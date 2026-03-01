

from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import time
import json
import re

from langchain_google_genai import GoogleGenerativeAI
from audio_utils import tts_enqueue_chunk
from concurrent.futures import ThreadPoolExecutor
import traceback
import mediapipe as mp
load_dotenv()
import cv2
from typing import Dict
from camera_utils import _calculate_angle, _classify_from_landmarks, _make_unknown_data

_tts_executor = ThreadPoolExecutor(max_workers=3)

class publicAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], ...]
    posture_history: list
    start_time: float
    evaluation_done: bool
    pass_meter: int
    turns: int
    last_question: str
    last_user_response: str
    last_transcript: str
    max_turns: int

llm = GoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
isPPT = True
mp_pose = mp.solutions.pose  
STALE_SECONDS=20
RANDOM_VOICES = ["p225", "p227", "p239"]
def _msg_content(m):
    if isinstance(m, BaseMessage):
        return getattr(m, "content", "") or ""
    if isinstance(m, dict):
        return m.get("content") or m.get("text") or m.get("body") or ""
    return str(m) if m is not None else ""

def _find_first_human(msgs):
    for m in msgs:
        if isinstance(m, HumanMessage):
            return m
        if isinstance(m, dict):
            typ = (m.get("type") or m.get("_type") or m.get("message_type") or "").lower()
            if "human" in typ or "user" in typ:
                return HumanMessage(content=_msg_content(m))
    return None

def llm_node(state: publicAgentState) -> publicAgentState:
    messages = list(state.get("messages", []))
    current_pass_meter = state.get("pass_meter", 0)
    meeting_topic = state.get("meeting_topic", "the presentation")
    first_human_msg = _find_first_human(messages)
    presentation = _msg_content(first_human_msg) if first_human_msg is not None else ""
    system_instruction = (
        "You are an audience member in a public presentation. OUTPUT EXACTLY ONE QUESTION ONLY on the future of artificial intelligence. "
        "One sentence, <=25 words, end with a question mark, nothing else. "
        "Ask questions about different aspects of the original presentation topic. "
        "Don't just ask follow-ups to the user's last response. "
        "If you cannot think of a relevant question, output exactly: 'Can you give one specific example?'\n"
        f"The presentation topic is: {meeting_topic}\n"
    )

    pass_meter_context = f"\nCurrent user performance score: {current_pass_meter}."
    if current_pass_meter <= -4:
        pass_meter_context += " The user is performing very poorly; be dismissive (still only one question)."
    elif current_pass_meter <= -2:
        pass_meter_context += " The user is performing poorly; be uninterested (still only one question)."
    elif current_pass_meter <= 0:
        pass_meter_context += " The user is neutral; be curious (1 short question)."
    else:
        pass_meter_context += " The user is performing well; be active and respectful (1 short question)."

    system_prompt = system_instruction + pass_meter_context

    history = [m for m in messages if not isinstance(m, SystemMessage)]
    modified_messages = [SystemMessage(content=system_prompt)] + history

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
            try:
                resp = llm.invoke(messages=modified_messages)
            except TypeError:
                resp = llm.invoke(modified_messages)
            candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
            assistant_text = getattr(candidate, "content", str(candidate)).strip()
            if "Evaluation:" in assistant_text:
                assistant_text = assistant_text.split("Evaluation:")[0].strip()
            assistant_text_accum = re.sub(r"\s+", " ", assistant_text).strip()
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
            assistant_text_accum = re.sub(r"\s+", " ", assistant_text).strip()
            try:
                _tts_executor.submit(tts_enqueue_chunk, assistant_text_accum)
            except Exception:
                pass
        except Exception:
            assistant_text_accum = "[Error generating assistant response]"

    assistant_text = re.sub(r"\s+", " ", assistant_text_accum).strip()

    def fallback_question() -> str:
        return "Can you give one specific example?"

    q_match = re.search(r'([^?]*\?)', assistant_text)
    first_question = q_match.group(1).strip() if q_match else None

    accept = False
    if first_question:
        word_count = len(first_question.split())
        if word_count <= 25:
            accept = True
            assistant_text = first_question
    else:
        if assistant_text.endswith('?') and len(assistant_text.split()) <= 25:
            accept = True

    if not accept:
        assistant_text = fallback_question()

    if not assistant_text.endswith('?'):
        assistant_text = assistant_text.rstrip('.') + '?'

    words = assistant_text.split()
    if len(words) > 25:
        assistant_text = ' '.join(words[:25]) + '?'

    ai_msg = AIMessage(content=assistant_text)

    return {
        "messages": messages + [ai_msg],
        "posture_history": state.get("posture_history", []),
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": current_pass_meter,
        "turns": state.get("turns", 0),
        "last_question": assistant_text,
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
        "meeting_topic": state.get("meeting_topic", ""),
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
        "messages": state["messages"],
        "posture_history": state["posture_history"],
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": state.get("pass_meter", 0),
        "turns": state.get("turns", 0),
        "last_question": state.get("last_question", ""),
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
        
    }


def evaluation_node(state: publicAgentState) -> publicAgentState:
    msgs = list(state["messages"])
    posture_data = state.get("posture_history", [])
    current_pass_meter = state.get("pass_meter", 0)

    last_human_msg = next(
        (m for m in reversed(msgs) if isinstance(m, HumanMessage) and hasattr(m, "content")),
        None
    )
    conversation_text = f"HUMAN: {last_human_msg.content}" if last_human_msg else ""

    evaluation_prompt_text = f"""You are an objective evaluator of public speaking delivery.
Latest response (only human line shown):
{conversation_text}
Posture History:
{posture_data}
Scoring rubric (delivery only):
- Provide a numeric score from 0 to 100 (0 worst, 100 best).
- Score is composed of attention (25), clarity (25), grammar (25), persuasiveness (25).
Return EXACTLY one JSON object and nothing else, like:
{{"decision":"PASS" or "FAIL", "score": 0-100, "explanation": "brief (max 200 chars) about delivery"}}
Decision rule: PASS if score >= 60, FAIL otherwise.
"""
    try:
        resp = llm.invoke(messages=[HumanMessage(content=evaluation_prompt_text)])
    except TypeError:
        resp = llm.invoke([HumanMessage(content=evaluation_prompt_text)])
    candidate = resp[0] if isinstance(resp, (list, tuple)) and resp else resp
    assistant_text = getattr(candidate, "content", str(candidate)).strip()
    json_match = re.search(r'(\{.*\})', assistant_text, re.DOTALL)

    decision = "FAIL"
    explanation = ""
    score = 0
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            score = int(parsed.get("score", 0))
            decision = parsed.get("decision", "FAIL").strip().upper()
            explanation = parsed.get("explanation", "")[:200]
        except Exception:
            decision = "FAIL"
            explanation = "Could not parse evaluator JSON."
    else:
        decision = "FAIL"
        explanation = assistant_text.replace("\n", " ")[:200]

    if decision == "PASS":
        new_pass_meter = current_pass_meter + 1
    else:
        new_pass_meter = current_pass_meter - 1

    new_turns = state.get("turns", 0) + 1
    print(f"Evaluation: {decision}. score={score}. {explanation}")

    return {
        "messages": msgs,
        "posture_history": posture_data,
        "start_time": state.get("start_time", time.time()),
        "evaluation_done": state.get("evaluation_done", False),
        "pass_meter": new_pass_meter,
        "turns": new_turns,
        "last_question": state.get("last_question", ""),
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
    }

def final_evaluation(state: publicAgentState) -> publicAgentState:
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
        "pass_meter": current_pass_meter,
        "turns": state.get("turns", 0),
        "last_question": state.get("last_question", ""),
        "last_user_response": state.get("last_user_response", ""),
        "last_transcript": state.get("last_transcript", ""),
        "max_turns": state.get("max_turns", 10),
    }

public_graph = StateGraph(publicAgentState)
public_graph.add_node("camera", posture_info_node)
public_graph.add_node("llm", llm_node)
public_graph.add_node("evaluation", evaluation_node)
public_graph.add_node("final_evaluation", final_evaluation)

public_graph.add_edge(START, "camera")
public_graph.add_edge("camera", "llm")
public_graph.add_edge("llm", "evaluation")
public_graph.add_edge("evaluation", "final_evaluation")
public_graph.add_edge("final_evaluation", END)
