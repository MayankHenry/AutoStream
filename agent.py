import os
import json
import requests
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from rag import setup_retriever

load_dotenv()

# ── 1. State ──────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    user_name: str
    user_email: str
    user_platform: str

# ── 2. LLM (OpenRouter) ───────────────────────────────────────────────────────

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/auto",
    temperature=0,
)

# ── 3. Mock Lead Capture Tool ─────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    print(f"\n🚀 [LEAD CAPTURED] Name: {name} | Email: {email} | Platform: {platform}\n")
    return f"Success! Lead captured for {name} ({email}) on {platform}."

# ── 4. Nodes ──────────────────────────────────────────────────────────────────

def detect_intent(state: AgentState) -> dict:
    """Classifies the latest user message into one of three intents."""
    messages = state.get("messages", [])

    system_prompt = """You are an intent classifier for AutoStream, a video editing SaaS.
Classify the user's LATEST message into EXACTLY ONE of:
  greeting     - casual hello, small talk
  inquiry      - questions about pricing, features, or policies
  high_intent  - user wants to buy, sign up, or try a plan

Reply with ONLY the single word: greeting, inquiry, or high_intent."""

    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    intent = response.content.strip().lower()

    if intent not in ("greeting", "inquiry", "high_intent"):
        intent = "greeting"

    print(f"[DEBUG] Intent: {intent}")
    return {"intent": intent}


def handle_greeting(state: AgentState) -> dict:
    """Handles casual greetings."""
    response = llm.invoke([
        SystemMessage(content=(
            "You are a friendly AI assistant for AutoStream, a video editing SaaS. "
            "Respond warmly and briefly (1-2 sentences). "
            "Mention you can help with pricing, features, or sign-up."
        ))
    ] + state["messages"])
    return {"messages": [response]}


retriever = setup_retriever()


def handle_inquiry(state: AgentState) -> dict:
    """Answers product/pricing questions using RAG."""
    user_query = state["messages"][-1].content

    docs = retriever.invoke(user_query)
    context = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""You are a helpful support agent for AutoStream.
Answer the user's question using ONLY the context below.
If the answer isn't in the context, say you don't know.
Keep your reply conversational and concise.

Context:
{context}"""

    response = llm.invoke([SystemMessage(content=rag_prompt)] + state["messages"])
    return {"messages": [response]}


def process_high_intent(state: AgentState) -> dict:
    """Collects lead info turn-by-turn and fires mock_lead_capture when complete."""
    latest_msg = state["messages"][-1].content

    # Try to extract any new info from the latest message
    extract_prompt = """Extract Name, Email, and Creator Platform from the message.
Return ONLY valid JSON: {"name": "", "email": "", "platform": ""}
Use empty string "" for any field not found. No markdown, no extra text."""

    extraction = llm.invoke([
        SystemMessage(content=extract_prompt),
        HumanMessage(content=latest_msg)
    ])

    try:
        cleaned = extraction.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
    except Exception:
        data = {"name": "", "email": "", "platform": ""}

    # Merge with previously collected state
    name     = data.get("name")     or state.get("user_name", "")
    email    = data.get("email")    or state.get("user_email", "")
    platform = data.get("platform") or state.get("user_platform", "")

    # Ask for the first missing field
    missing = []
    if not name:     missing.append("full name")
    if not email:    missing.append("email address")
    if not platform: missing.append("creator platform (e.g. YouTube, Instagram)")

    if missing:
        reply = (
            f"I'd love to get you started on the Pro plan! "
            f"Could I get your {missing[0]}?"
        )
    else:
        result = mock_lead_capture(name, email, platform)
        reply = f"{result} Our team will reach out to you shortly. Welcome to AutoStream! 🎬"

    return {
        "messages": [reply],
        "user_name": name,
        "user_email": email,
        "user_platform": platform,
    }

# ── 5. Router ─────────────────────────────────────────────────────────────────

def route_by_intent(state: AgentState) -> str:
    intent = state.get("intent", "greeting")
    return {
        "greeting":    "handle_greeting",
        "inquiry":     "handle_inquiry",
        "high_intent": "process_high_intent",
    }.get(intent, "handle_greeting")

# ── 6. Graph ──────────────────────────────────────────────────────────────────

workflow = StateGraph(AgentState)

workflow.add_node("detect_intent",      detect_intent)
workflow.add_node("handle_greeting",    handle_greeting)
workflow.add_node("handle_inquiry",     handle_inquiry)
workflow.add_node("process_high_intent", process_high_intent)

workflow.add_edge(START, "detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    route_by_intent,
    {
        "handle_greeting":    "handle_greeting",
        "handle_inquiry":     "handle_inquiry",
        "process_high_intent": "process_high_intent",
    }
)
workflow.add_edge("handle_greeting",    END)
workflow.add_edge("handle_inquiry",     END)
workflow.add_edge("process_high_intent", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)