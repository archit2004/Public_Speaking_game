
#imports
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
import os
import time
import json
import re
from audio_utils import record_audio, speech_to_text, text_to_speech
from camera_utils import _calculate_angle,_classify_from_landmarks, _make_unknown_data
from langchain_google_genai import GoogleGenerativeAI
from audio_utils import record_audio, speech_to_text, text_to_speech, tts_enqueue_chunk
import traceback
from concurrent.futures import ThreadPoolExecutor
import traceback
import time
import cv2
import mediapipe as mp
from typing import Dict
_tts_executor = ThreadPoolExecutor(max_workers=3)

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]
    posture_history: list
    start_time: float
    evaluation_done: bool
    pass_meter: int
llm=GoogleGenerativeAI(model="gemini-2.5-flash",temperature=0.6)




mp_pose = mp.solutions.pose  
STALE_SECONDS=20
BOSS_FEMALE = "p231"
FRIENDLY_COWORKER = "p248"
def llm_node(state: AgentState) -> AgentState:
    messages = list(state["messages"])
    current_pass_meter = state.get("pass_meter", 0)

    if messages and getattr(messages[-1], "content", "").strip().lower() == "exit":
        messages.append(SystemMessage(content="Goodbye!"))
        return {
            "messages": messages,
            "posture_history": state.get("posture_history", []),
            "start_time": state.get("start_time", time.time()),
            "evaluation_done": True,
            "pass_meter": current_pass_meter
        }

    pass_meter_context = f"\n\nCurrent user performance score: {current_pass_meter}. "
    if current_pass_meter <= -4:
        pass_meter_context += "The user is performing very poorly. Be extremely rude, dismissive, and impatient. Use short, harsh responses. Do not elaborate."
    elif current_pass_meter <= -2:
        pass_meter_context += "The user is performing poorly. Be critical and skeptical. Use short responses. Challenge every point."
    elif current_pass_meter <= 0:
        pass_meter_context += "The user is performing neutrally. Be professional but challenging. Use concise responses."
    else:
        pass_meter_context += "The user is performing well. Show grudging respect but still push back. Use concise responses."

    modified_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            modified_messages.append(SystemMessage(content=msg.content + pass_meter_context))
        else:
            modified_messages.append(msg)

    print("Sending messages to LLM (history length =", len(modified_messages), ")")

    assistant_text_accum = ""
    try:
        stream_generator = None
        if hasattr(llm, "stream"):
            try:
                stream_generator = llm.stream(messages=modified_messages)
            except Exception:
                try:
                    stream_generator = llm.stream(modified_messages)
                except Exception:
                    stream_generator = None

        if stream_generator is None and hasattr(llm, "invoke"):
            try:
                stream_generator = llm.invoke(messages=modified_messages, stream=True)
            except Exception:
                try:
                    stream_generator = llm.invoke(modified_messages, stream=True)
                except Exception:
                    stream_generator = None

        if stream_generator is None:
            print("LLM streaming not available; using synchronous invoke.")
            try:
                resp = llm.invoke(messages=modified_messages)
            except TypeError:
                resp = llm.invoke(modified_messages)
            candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
            assistant_text = getattr(candidate, "content", str(candidate))
            if "Evaluation:" in assistant_text:
                assistant_text = assistant_text.split("Evaluation:")[0].strip()
            assistant_text_accum = assistant_text
            try:
                _tts_executor.submit(tts_enqueue_chunk, assistant_text_accum)
            except Exception:
                print("Failed to submit one-shot TTS task.")
        else:
            buffer_for_tts = ""
            flush_threshold_chars = 220  
            min_chunk_chars = 30      
            punctuation_triggers = {".", "!", "?", "\n"}

            for chunk in stream_generator:
                print("STREAM CHUNK:", repr(chunk))
                token = None
                if hasattr(chunk, "content"):
                    token = getattr(chunk, "content", None)
                    if token is None or token == "":
                        continue
                elif isinstance(chunk, dict):
                    token = chunk.get("content") or chunk.get("delta") or chunk.get("text") or None
                    if not token:
                        continue
                elif isinstance(chunk, str):
                    token = chunk
                else:
                    s = str(chunk).strip()
                    token = s if s else None
                    if not token:
                        continue
                assistant_text_accum += token
                buffer_for_tts += token
                if "Evaluation:" in assistant_text_accum:
                    assistant_text_accum = assistant_text_accum.split("Evaluation:")[0].strip()
                    if len(buffer_for_tts.strip()) >= min_chunk_chars:
                        try:
                            _tts_executor.submit(tts_enqueue_chunk, buffer_for_tts)
                        except Exception:
                            print("TTS submission error on evaluation flush:", traceback.format_exc())
                    buffer_for_tts = ""
                    break

                should_flush = False
                if any(buffer_for_tts.strip().endswith(p) for p in punctuation_triggers) and len(buffer_for_tts.strip()) >= min_chunk_chars:
                    should_flush = True
                elif len(buffer_for_tts) >= flush_threshold_chars:
                    take_up_to = None
                    snippet = buffer_for_tts[:flush_threshold_chars]
                    last_space = snippet.rfind(" ")
                    if last_space > 0:
                        take_up_to = last_space
                    else:
                        if len(buffer_for_tts) >= (flush_threshold_chars + 10):
                            take_up_to = flush_threshold_chars
                    if take_up_to:
                        chunk_text = buffer_for_tts[:take_up_to]
                        buffer_for_tts = buffer_for_tts[take_up_to:].lstrip()
                        if len(chunk_text.strip()) >= min_chunk_chars:
                            try:
                                _tts_executor.submit(tts_enqueue_chunk, chunk_text)
                            except Exception:
                                print("TTS submit error (threshold flush):", traceback.format_exc())
                        
                        continue

                if should_flush:
                    chunk_text = buffer_for_tts
                    buffer_for_tts = ""
                    if len(chunk_text.strip()) >= min_chunk_chars:
                        try:
                            _tts_executor.submit(tts_enqueue_chunk, chunk_text)
                        except Exception:
                            print("TTS submit error (punctuation flush):", traceback.format_exc())

            if buffer_for_tts.strip() and len(buffer_for_tts.strip()) >= min_chunk_chars:
                try:
                    _tts_executor.submit(tts_enqueue_chunk, buffer_for_tts)
                except Exception:
                    print("TTS submit error (final flush):", traceback.format_exc())

    except Exception as e:
        print("Error during streaming/invoke:", e)
        print(traceback.format_exc())
        try:
            resp = llm.invoke(messages=modified_messages)
            candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
            assistant_text = getattr(candidate, "content", str(candidate))
            if "Evaluation:" in assistant_text:
                assistant_text = assistant_text.split("Evaluation:")[0].strip()
            assistant_text_accum = assistant_text
            try:
                _tts_executor.submit(tts_enqueue_chunk, assistant_text_accum)
            except Exception:
                pass
        except Exception:
            assistant_text_accum = "[Error generating assistant response]"

    ai_msg = AIMessage(content=assistant_text_accum)

    return {
        "messages": messages + [ai_msg],
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": current_pass_meter
    }






def posture_info_node(state: Dict) -> Dict:
    """
    Single-tick node: grab one frame (if camera is available), run mediapipe pose once,
    compute posture, and update state. No UDP involved.
    """
    state.setdefault("posture_history", [])
    state.setdefault("messages", [])
    state.setdefault("audio_history", [])
    state.setdefault("start_time", time.time())

    if "posture_cap" not in state:
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            state["posture_cap"] = cap
            state["mp_pose_session"] = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print(f"[posture_info_node] camera initialized")
        except Exception as e:
            print(f"[posture_info_node] camera init failed: {e}")
            state["posture_cap"] = None
            state["mp_pose_session"] = None

    cap = state.get("posture_cap")
    pose_session = state.get("mp_pose_session")
    data = state.get("last_posture", _make_unknown_data())

    received_any = False

    if cap is None or not cap.isOpened() or pose_session is None:
 
        print("[posture_info_node] camera not available this tick")
    else:
        ret, frame = cap.read()
        if not ret:
            print("[posture_info_node] camera read failed this tick")
        else:
          
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose_session.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    payload, angle = _classify_from_landmarks(landmarks)
                    now = time.time()
                    if payload == "upright":
                        data = {
                            "posture": "upright",
                            "confidence": 1.0,
                            "source": "camera",
                            "raw_text": f"{payload} ({angle:.1f}°)",
                            "received_time": now,
                        }
                    else:
                        data = {
                            "posture": "slouched",
                            "confidence": None,
                            "source": "camera",
                            "raw_text": f"{payload} ({angle:.1f}°)",
                            "received_time": now,
                        }
                    state["last_posture"] = data
                    received_any = True
                    print(f"[posture_info_node] computed posture: {data['raw_text']}")
                except Exception as e:
                    print("[posture_info_node] processing error:", repr(e))
            else:
               
                print("[posture_info_node] no landmarks detected this tick")

    if received_any:
        state.setdefault("posture_history", []).append(state["last_posture"])
    else:
  
        last = state.get("last_posture")
        if last and (time.time() - last.get("received_time", 0.0) < STALE_SECONDS):
            state.setdefault("posture_history", []).append(last)
            print(f"[posture_info_node] (fresh) {last}")
        else:
            pass

    return {
        "messages": state.get("messages", []),
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": state.get("pass_meter", 0),
    }





def evaluation_node(state: AgentState) -> AgentState:
    msgs = list(state["messages"])
    posture_data = state.get("posture_history", [])
    current_pass_meter = state.get("pass_meter", 0)

    last_humans = [m for m in reversed(msgs) if isinstance(m, HumanMessage)][:2]
    last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)

    selected_msgs = list(reversed(last_humans))
    if last_ai:
        selected_msgs.append(last_ai)

    conversation_text = "\n".join(
        f"{m.type.upper()}: {m.content}"
        for m in selected_msgs
        if hasattr(m, "content") and m.content
    )

    summary = "Posture and Confidence History:\n"
    for entry in posture_data:
        summary += (
            f"- Posture: {entry.get('posture')}, Gaze: {entry.get('gaze')}, "
            f"Confidence: {entry.get('confidence')}, Arms: {entry.get('arms')}, "
            f"Head Tilt: {entry.get('head_tilt')}\n"
        )

    evaluation_prompt_text = f"""You are evaluating a public speaking performance in a training exercise.

Conversation (last 2 Human msgs + last AI msg):
{conversation_text}
Posture History:
{posture_data}

Evaluation criteria (focus on delivery, not content):

1. (25%)Based on the response, did the user appear calm and confident?
2. (25%)Was the speech clear and easy to understand?
3. (25%)Was the grammar correct?
4. (25%)Did the user make a persuasive argument?
5. DO NOT focus on the actual content of the presentation but rather on the public speaking skills.

Scoring:
- PASS if the speaker demonstrates good public speaking skills (score ≥60%)
- FAIL if the speaker needs significant improvement (score <60%)

Return EXACTLY one JSON object with two fields:
{{"decision": "PASS" or "FAIL", "explanation": "brief explanation focusing on delivery skills"}}
"""

    human_msg = HumanMessage(content=evaluation_prompt_text)

    try:
        resp = llm.invoke(messages=[human_msg])
    except TypeError:
        resp = llm.invoke([human_msg])

    candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
    assistant_text = getattr(candidate, "content", str(candidate)).strip()

    decision = None
    explanation = None
    try:
        parsed = json.loads(assistant_text)
        decision = parsed.get("decision", "").strip().upper()
        explanation = parsed.get("explanation", "").strip()
    except Exception:
        m = re.search(r'\b(PASS|FAIL)\b[:\-\s]*(.*)', assistant_text, re.IGNORECASE)
        if m:
            decision = m.group(1).upper()
            explanation = m.group(2).strip()[:200] if m.group(2) else ""
        else:
            decision = "FAIL"
            explanation = assistant_text.replace("\n", " ")[:200]

    if decision == "PASS":
        new_pass_meter = current_pass_meter + 2
        print("✅ pass_meter increased to:", new_pass_meter)
    else:
        new_pass_meter = current_pass_meter - 2
        print("❌ pass_meter decreased to:", new_pass_meter)

    print(f"Evaluation: {decision}. {explanation}")

    return {
        "messages": msgs,
        "posture_history": posture_data,
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": new_pass_meter
    }


def final_evaluation(state: AgentState) -> AgentState:
    current_pass_meter = state.get("pass_meter", 0)
    print(f"Pass meter final value: {current_pass_meter}")
    if current_pass_meter >= 0:
        print("Passed")
    else:
        print("Failed")
    return {
        "messages": state["messages"],
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": True,
        "pass_meter": current_pass_meter
    }


#Graph
graph = StateGraph(AgentState)
graph.add_node("camera", posture_info_node)
graph.add_node("llm", llm_node)
graph.add_node("evaluation", evaluation_node)

# edges
graph.add_edge(START, "camera")
graph.add_edge("camera", "llm")
graph.add_edge("llm", "evaluation")
graph.add_edge("evaluation", END)

