from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

from audio_utils import record_audio, speech_to_text, text_to_speech
from camera_utils import detect_posture_and_confidence

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

def stt_node(state: AgentState) -> AgentState:
    audio_file = record_audio()
    text = speech_to_text(audio_file)
    
    try:
        os.remove(audio_file)
    except:
        pass
        
    if "exit" in text.lower() or "quit" in text.lower() or "stop" in text.lower():
        return {"messages": list(state.get("messages", [])) + [HumanMessage(content="exit")]}
    
    human_msg = HumanMessage(content=text)
    new_messages = list(state.get("messages", [])) + [human_msg]
    return {"messages": new_messages}

def llm_node(state: AgentState) -> AgentState:
    messages = list(state.get("messages", []))
    if messages and hasattr(messages[-1], 'content') and messages[-1].content.lower() == "exit":
        return {"messages": messages + [SystemMessage(content="Goodbye!")]}
    
    print("Sending messages to LLM (history length =", len(messages), ")")
    try:
        resp = llm.invoke(messages=messages)
    except TypeError:
        resp = llm.invoke(messages)

    if isinstance(resp, (list, tuple)) and resp:
        candidate = resp[0]
    else:
        candidate = resp

    assistant_text = getattr(candidate, "content", str(candidate))
    ai_msg = AIMessage(content=assistant_text)
    new_messages = messages + [ai_msg]
    return {"messages": new_messages}

def tts_node(state: AgentState) -> AgentState:
    msgs = list(state.get("messages", []))
    if not msgs:
        return state
        
    last = msgs[-1]
    text = getattr(last, "content", str(last))
    
    if text.lower() in ["exit", "quit", "stop", "goodbye!"]:
        return state
        
    text_to_speech(text)
    return {"messages": msgs}

def continue_conv(state: AgentState) -> str:
    messages = list(state.get("messages", []))
    if messages and hasattr(messages[-1], 'content'):
        if messages[-1].content.lower() == "exit":
            return "end"
        if isinstance(messages[-1], SystemMessage) and "goodbye" in messages[-1].content.lower():
            return "end"
    return "continue"




def posture_info_node(state):
    if "posture_history" not in state:
        state["posture_history"] = []

    data = detect_posture_and_confidence()

    # Save into history (dict)
    state["posture_history"].append(data)

    print(
        f"[Posture Info] Posture: {data['posture']}, Gaze: {data['gaze']}, "
        f"Confidence: {data['confidence']}, Arms: {data['arms']}, "
        f"Head tilt: {data['head_tilt']}"
    )

    if data["confidence"] == "Confident":
        nudge_msg = "You're doing well! Keep up your confident posture."
    else:
        nudge_msg = "Try to sit upright, open your arms, and look at the camera."

    msg_text = (
        f"Posture: {data['posture']}, Gaze: {data['gaze']}, "
        f"Confidence: {data['confidence']}, Arms: {data['arms']}, "
        f"Head Tilt: {data['head_tilt']}\nNudge: {nudge_msg}"
    )
    return SystemMessage(content=msg_text)

graph = StateGraph(AgentState)
graph.add_node("stt", stt_node)
graph.add_node("llm", llm_node)
graph.add_node("tts", tts_node)
graph.add_node("camera", posture_info_node)

graph.add_edge(START, "stt")
graph.add_edge("stt", "camera")
graph.add_edge("camera", "llm")
graph.add_edge("llm", "tts")

graph.add_conditional_edges(
    "tts",
    continue_conv,
    {
        "continue": "stt",
        "end": END,
    },
)

app = graph.compile()

if __name__ == "__main__":
    seed: AgentState = {
        "messages": 
        [SystemMessage(content="You are a coach to help users boost their confidence help them with public speaking. Monitor their posture and conversation throughout you conversation and give comments regarding it. You will be given posture, wether their gaze is looking at the camera and wether their confident based on their camera along with the human message. Help them based only on that. Try to get improve their confidence through this conversation")]}
    app.invoke(seed)
    print("Conversation finished.")