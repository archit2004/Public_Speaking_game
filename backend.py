


from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import time
from langchain_ollama import ChatOllama
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from convince_boss import graph, AgentState
from public_speaking import publicAgentState, public_graph
import tempfile
import threading
import uuid
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from audio_utils import speech_to_text, text_to_speech, speak
import asyncio, aiofiles
from concurrent.futures import ThreadPoolExecutor
load_dotenv()
app = FastAPI()

_SESSION_DB: dict = {}           
_ENDPOINT_THREAD_MAP: dict = {}  
DB_EXPIRY_SECONDS = 120.0        


def _get_or_create_session_db(thread_key: str, prefix: str, force_new: bool = False) -> str:
    now = time.time()
    entry = _SESSION_DB.get(thread_key)

    if not force_new and entry:
        path, created = entry
        if (now - created) < DB_EXPIRY_SECONDS and os.path.exists(path):
            return path
        try:
            os.remove(path)
        except Exception:
            pass
        _SESSION_DB.pop(thread_key, None)

    fd, path = tempfile.mkstemp(prefix=f"{prefix}_{thread_key}_", suffix=".sqlite")
    os.close(fd)
    _SESSION_DB[thread_key] = (path, now)

    def _cleanup(p=path, k=thread_key):
        try:
            os.remove(p)
        except Exception:
            pass
        _SESSION_DB.pop(k, None)

    t = threading.Timer(DB_EXPIRY_SECONDS, _cleanup)
    t.daemon = True
    t.start()
    return path

def _delete_session_db(thread_key: str):
    entry = _SESSION_DB.pop(thread_key, None)
    if entry:
        path, _ = entry
        try:
            os.remove(path)
        except Exception:
            pass
    for ep, (tk, _) in list(_ENDPOINT_THREAD_MAP.items()):
        if tk == thread_key:
            _ENDPOINT_THREAD_MAP.pop(ep, None)

def _ensure_thread_key_for_endpoint(endpoint: str, max_age: float = DB_EXPIRY_SECONDS) -> str:
    now = time.time()
    entry = _ENDPOINT_THREAD_MAP.get(endpoint)
    if entry:
        thread_key, created = entry
        session_entry = _SESSION_DB.get(thread_key)
        if session_entry:
            _, session_created = session_entry
            if (now - session_created) < max_age and os.path.exists(session_entry[0]):
                return thread_key
    thread_key = f"{endpoint}_{uuid.uuid4().hex[:8]}"
    _ENDPOINT_THREAD_MAP[endpoint] = (thread_key, now)
    return thread_key

def _get_db_for_endpoint(endpoint: str, prefix: str, force_new: bool = False):
    thread_key = _ensure_thread_key_for_endpoint(endpoint)
    if force_new:
        _delete_session_db(thread_key)
    db_path = _get_or_create_session_db(thread_key, prefix, force_new=False)
    config = {"configurable": {"thread_id": thread_key}}
    return db_path, config, thread_key

def _reconstruct_message(m):
    if isinstance(m, BaseMessage):
        return m
    if isinstance(m, dict):
        typ = (m.get("type") or m.get("_type") or m.get("message_type") or "").lower()
        content = m.get("content") or m.get("text") or m.get("body") or ""
        if "system" in typ:
            return SystemMessage(content=content or "")
        if "human" in typ or "user" in typ:
            return HumanMessage(content=content or "")
        if "ai" in typ or "assistant" in typ:
            return AIMessage(content=content or "")
    return m

def _msg_content(m):
    if isinstance(m, BaseMessage):
        return getattr(m, "content", "") or ""
    if isinstance(m, dict):
        return m.get("content") or m.get("text") or m.get("body") or ""
    return str(m) if m is not None else ""

def _last_ai_or_assistant_content(msgs):
    for m in reversed(msgs or []):
        if isinstance(m, AIMessage):
            return getattr(m, "content", "")
        if isinstance(m, dict):
            typ = (m.get("type") or m.get("_type") or m.get("message_type") or "").lower()
            if "ai" in typ or "assistant" in typ:
                return m.get("content") or m.get("text") or ""
    return ""

def _first_system_message_text(msgs):
    for m in msgs or []:
        if isinstance(m, SystemMessage):
            return getattr(m, "content", "") or ""
        if isinstance(m, dict):
            typ = (m.get("type") or m.get("_type") or m.get("message_type") or "").lower()
            if "system" in typ:
                return m.get("content") or m.get("text") or ""
    return ""

llm = ChatOllama(model="mistral:instruct", temperature=0.0)
_tts_executor = ThreadPoolExecutor(max_workers=3)


@app.post("/convince_boss")
async def convince_boss(audioFile: UploadFile = File(...)):
    meeting_topic = "Future of Artificial Intelligence?"
    contents = await audioFile.read()
    text = speech_to_text(contents)

    seed: AgentState = {
        "messages": [
            SystemMessage(content=f"""
You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
You are playing the role of the boss of the user. 
The user has to try to convince you to let them present in an important meeting. 
The topic of the meeting is '{meeting_topic}'.

Instructions for the roleplay:
- Your tone should adapt based on the user's performance (pass_meter value)
- Respond only with dialogue, as if you are speaking directly to the user.  
- Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
- Be firm and direct. Give constructive, actionable feedback in short sentences.
- Push back, challenge their arguments, and make them defend themselves.  
- Keep your responses very short and to the point — no more than 1-2 sentences.
- If the user's performance is poor (negative pass_meter), be increasingly rude and dismissive.
- Use 1 or 2 short sentences only in your response.
"""), HumanMessage(content=text),
        ],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 2
    }

    DB_PATH_CONV, config_conv, thread_key_conv = _get_db_for_endpoint("convince_boss", "checkpointsconv")

    with SqliteSaver.from_conn_string(DB_PATH_CONV) as memory:
        graph_app = graph.compile(checkpointer=memory)
        snapshot = memory.get(config_conv)

        if snapshot:
            previous_state = dict(snapshot.get("channel_values", {}))
            msgs = previous_state.get("messages", [])
            msgs = [_reconstruct_message(m) for m in msgs]

            first_system = _first_system_message_text(msgs)
            if first_system and "role of the boss" not in first_system.lower() and "boss" in first_system.lower() == False:
            
                print("[convince_boss] snapshot role mismatch -> resetting session DB")
                _delete_session_db(thread_key_conv)
                DB_PATH_CONV, config_conv, thread_key_conv = _get_db_for_endpoint("convince_boss", "checkpointsconv", force_new=True)
                with SqliteSaver.from_conn_string(DB_PATH_CONV) as memory2:
                    graph_app = graph.compile(checkpointer=memory2)
                    final_state = graph_app.invoke(seed, config_conv)
            else:
                msgs.append(HumanMessage(content=text))
                previous_state["messages"] = msgs

                start_time = previous_state.get("start_time", seed["start_time"])
                elapsed = time.time() - start_time
                if elapsed > DB_EXPIRY_SECONDS:
                    final_pass_meter = previous_state.get("pass_meter", 0)
                    timeout_msg = f"I'm out of time. Final pass meter: {final_pass_meter}"
                    print("[convince_boss] TIMEOUT:", timeout_msg)
                    try:
                        text_to_speech(timeout_msg)
                    except Exception as e:
                        print("[convince_boss] TTS on timeout failed:", e)
                    success = final_pass_meter >= 0
                    return {
                        "text_response": timeout_msg,
                        "sucess": success,
                        "pass_meter": final_pass_meter
                    }

                final_state = graph_app.invoke(previous_state, config_conv)
        else:
            print("[convince_boss] Starting new conversation...")
            final_state = graph_app.invoke(seed, config_conv)

        boss_reply = _last_ai_or_assistant_content(final_state.get("messages", []))

        final_pass_meter = final_state.get("pass_meter", 0)
        print(f"\n--- Final Result ---")
        print(f"Pass meter: {final_pass_meter}")
        success = final_pass_meter >= 0
        print("Overall:", "PASSED" if success else "FAILED")
        print("Conversation finished.")

        return {
            "text_response": boss_reply,
            "sucess": success,
            "pass_meter": final_pass_meter
        }

@app.post("/public_speaking")
async def public_speaking(audioFile: UploadFile = File(...)):
    meeting_topic = "Future of Artificial Intelligence?"
    max_interactions = 5  

    contents = await audioFile.read()
    text = speech_to_text(contents)
    public_seed: publicAgentState = {
        "messages": [],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 0,
        "turns": 0,
        "last_question": "",
        "last_user_response": "",
        "last_transcript": "",
        "max_turns": max_interactions,
        "meeting_topic": meeting_topic,
    }

    while True:
        DB_PATH_PUB, config_pub, thread_key_pub = _get_db_for_endpoint("public_speaking", "checkpointspub")
        with SqliteSaver.from_conn_string(DB_PATH_PUB) as memory:
            graph_app = public_graph.compile(checkpointer=memory)
            snapshot = memory.get(config_pub)

            if snapshot:
                prev = dict(snapshot.get("channel_values", {}))
                stored_turns = int(prev.get("turns", 0) or 0)
                if stored_turns >= max_interactions:
                    print(f"[public_speaking] session reached max turns ({stored_turns}); resetting session DB.")
                    _delete_session_db(thread_key_pub)
                    continue
            break

    DB_PATH_PUB, config_pub, thread_key_pub = _get_db_for_endpoint("public_speaking", "checkpointspub")
    with SqliteSaver.from_conn_string(DB_PATH_PUB) as memory:
        graph_app = public_graph.compile(checkpointer=memory)
        snapshot = memory.get(config_pub)

        if snapshot:
            previous_state = dict(snapshot.get("channel_values", {}))
            msgs = previous_state.get("messages", [])
            msgs = [_reconstruct_message(m) for m in msgs]
            msgs.append(HumanMessage(content=text))
            previous_state["messages"] = msgs

            current_turns = int(previous_state.get("turns", 0))
            if current_turns >= max_interactions:
                final_pass_meter = previous_state.get("pass_meter", 0)
                timeout_msg = f"Well thats it. Final pass meter: {final_pass_meter}"
                print("[public_speaking] MAX INTERACTIONS:", timeout_msg)
                try:
                    text_to_speech(timeout_msg)
                except Exception as e:
                    print("[public_speaking] TTS on max-interactions failed:", e)
                success = final_pass_meter >= 0
                return {
                    "text_response": timeout_msg,
                    "sucess": success,
                    "pass_meter": final_pass_meter
                }

            final_state = graph_app.invoke(previous_state, config_pub)
        else:
            print("[public_speaking] Starting new conversation...")
            final_state = graph_app.invoke(public_seed, config_pub)

        audience_reply = _last_ai_or_assistant_content(final_state.get("messages", []))

        final_pass_meter = final_state.get("pass_meter", 0)
        print(f"\n--- Final Result ---")
        print(f"Excitement meter: {final_pass_meter}")
        success = final_pass_meter >= 0
        print("Overall:", "PASSED" if success else "FAILED")
        print("Conversation finished.")

        return {
            "text_response": audience_reply,
            "sucess": success,
            "Excitement_meter": final_pass_meter
        }

@app.post("/play_npc")
async def play_npc(audioFile: UploadFile = File(...)):
    meeting_topic = "future of artificial intelligence?"
    contents = await audioFile.read()
    text = speech_to_text(contents)

    seed: AgentState = {
        "messages": [
            SystemMessage(content=f"""
You are an NPC in an educational video game to help young adults learn public speaking in the corporate world. 
You are playing the role of the coworker of the user. 
you have to give hints to the user regarding an important upcoming meeting. 
The topic of the meeting is '{meeting_topic}'.

Instructions for the roleplay:
- Your tone should adapt based on the user's performance (pass_meter value)
- Respond only with dialogue, as if you are speaking directly to the user.  
- Do NOT include stage directions, narration, or descriptions like (leans back) or *smiles*.  
- Be friendly and fun, but formal.
- Keep your responses very short and to the point — no more than 1-2 sentences.
- If the user's performance is poor (negative pass_meter), be increasingly rude and dismissive.
- Use 1 or 2 short sentences only in your response.
"""), HumanMessage(content=text),
        ],
        "posture_history": [],
        "evaluation_done": False,
        "start_time": time.time(),
        "pass_meter": 0
    }

    DB_PATH_NPC, config_npc, thread_key_npc = _get_db_for_endpoint("play_npc", "checkpointsnpc")

    with SqliteSaver.from_conn_string(DB_PATH_NPC) as memory:
        graph_app = graph.compile(checkpointer=memory)
        snapshot = memory.get(config_npc)

        if snapshot:
            previous_state = dict(snapshot.get("channel_values", {}))
            msgs = previous_state.get("messages", [])
            msgs = [_reconstruct_message(m) for m in msgs]

            first_system = _first_system_message_text(msgs)
            if first_system and ("boss of the user" in first_system.lower() or "role of the boss" in first_system.lower()):
                print("[play_npc] snapshot role mismatch (found boss content) -> resetting session DB")
                _delete_session_db(thread_key_npc)
                DB_PATH_NPC, config_npc, thread_key_npc = _get_db_for_endpoint("play_npc", "checkpointsnpc", force_new=True)
                with SqliteSaver.from_conn_string(DB_PATH_NPC) as memory2:
                    graph_app = graph.compile(checkpointer=memory2)
                    final_state = graph_app.invoke(seed, config_npc)
            else:
                msgs.append(HumanMessage(content=text))
                previous_state["messages"] = msgs

                start_time = previous_state.get("start_time", seed["start_time"])
                elapsed = time.time() - start_time
                if elapsed > DB_EXPIRY_SECONDS:
                    final_pass_meter = previous_state.get("pass_meter", 0)
                    timeout_msg = f"I have to run. See you later!"
                    print("[play_npc] TIMEOUT:", timeout_msg)
                    try:
                        text_to_speech(timeout_msg)
                    except Exception as e:
                        print("[play_npc] TTS on timeout failed:", e)
                    success = final_pass_meter >= 0
                    return {
                        "text_response": timeout_msg,
                        "sucess": success,
                        "pass_meter": final_pass_meter
                    }

                final_state = graph_app.invoke(previous_state, config_npc)
        else:
            print("[play_npc] Starting new conversation...")
            final_state = graph_app.invoke(seed, config_npc)

        npc_reply = _last_ai_or_assistant_content(final_state.get("messages", []))

        final_pass_meter = final_state.get("pass_meter", 0)
        print(f"\n--- Final Result ---")
        print(f"Pass meter: {final_pass_meter}")
        success = final_pass_meter >= 0

        return {
            "text_response": npc_reply,
            "sucess": success,
            "pass_meter": final_pass_meter
        }
