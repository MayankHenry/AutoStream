import os
import json
import requests
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# LangChain & OpenRouter Imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# LangGraph Imports
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Local Imports
from rag import setup_retriever

# Load your OpenRouter API key
load_dotenv("_env")

# ==========================================
# 1. Define the Agent's Memory (State)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list, add_messages] 
    intent: str 
    user_name: str
    user_email: str
    user_platform: str

# ==========================================
# 2. Initialize Gemini 1.5 Flash (Via OpenRouter)
# ==========================================
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
   model="google/gemini-2.0-flash-exp:free",
    temperature=0,
)

# ==========================================
# 3. Mock Tool Definition
# ==========================================
def mock_lead_capture(name, email, platform):
    print(f"\n🚀 [TOOL EXECUTION] Lead captured successfully: {name}, {email}, {platform}\n")
    return f"Success! {name} from {platform} has been captured as a lead."

# ==========================================
# 4. Agent Nodes (The Brain & Hands)
# ==========================================
def detect_intent(state: AgentState):
    """Analyzes the conversation and updates the intent state."""
    messages = state.get("messages", [])
    
    system_prompt = """You are the intent classification brain for AutoStream, a video editing SaaS.
    Analyze the conversation history and categorize the user's LATEST message into EXACTLY ONE of these three intents:
    
    1. 'greeting' (casual hellos, asking how you are)
    2. 'inquiry' (asking about pricing, features, or company policies)
    3. 'high_intent' (expressing desire to buy, sign up, or try a plan)
    
    Respond ONLY with the exact string: greeting, inquiry, or high_intent. Do not add punctuation or extra text."""
    
    response = llm.invoke([SystemMessage(content=system_prompt)] + messages)
    detected_intent = response.content.strip().lower()
    
    # Fallback to greeting if the LLM hallucinated formatting
    if detected_intent not in ['greeting', 'inquiry', 'high_intent']:
        detected_intent = 'greeting'
        
    print(f"\n[DEBUG] Intent Detected: {detected_intent}")
    return {"intent": detected_intent}

def handle_greeting(state: AgentState):
    """Responds to simple hellos without querying the database."""
    response = llm.invoke([
        SystemMessage(content="You are a helpful, friendly AI assistant for AutoStream. Keep your greeting brief (1-2 sentences).")
    ] + state["messages"])
    return {"messages": [response]}

# Initialize the retriever once
retriever = setup_retriever()

def handle_inquiry(state: AgentState):
    """Uses ChromaDB + Cohere Reranking to answer pricing and policy questions."""
    user_query = state["messages"][-1].content
    
    # 1. Broad Retrieval: Fetch top 5 chunks from ChromaDB
    retriever.search_kwargs = {"k": 5}
    initial_docs = retriever.invoke(user_query)
    doc_texts = [doc.page_content for doc in initial_docs]
    
    # 2. Rerank: Use OpenRouter and Cohere to sort them by actual relevance
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/rerank",
            headers={
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "cohere/rerank-4-pro",
                "query": user_query,
                "documents": doc_texts,
                "top_n": 2 # Only keep the absolute best 2 documents
            })
        )
        
        results = response.json()
        
        # Rebuild the context using the reranked indices
        best_chunks = [doc_texts[result['index']] for result in results.get("results", [])]
        context = "\n".join(best_chunks)
        print("\n[DEBUG] Advanced RAG: Documents successfully reranked by Cohere!")
        
    except Exception as e:
        print(f"\n[DEBUG] Reranking failed, falling back to standard RAG. Error: {e}")
        # Fallback to standard RAG if the API call fails
        context = "\n".join(doc_texts[:2])
    
    # 3. Prompt the LLM using ONLY the reranked context
    rag_prompt = f"""You are a support agent for AutoStream. 
    Answer the user's question accurately using ONLY the context provided below. 
    If the answer is not in the context, say you don't know. Keep it conversational.
    
    Context:
    {context}
    """
    
    response = llm.invoke([SystemMessage(content=rag_prompt)] + state["messages"])
    return {"messages": [response]}

def process_high_intent(state: AgentState):
    """Extracts lead info, checks state memory, and prompts for missing data."""
    latest_msg = state["messages"][-1].content
    
    extract_sys = """Extract Name, Email, and Platform (e.g., YouTube, Instagram) from the user's message.
    Respond STRICTLY in valid JSON format: {"name": "", "email": "", "platform": ""}.
    If a field is not mentioned, leave it as an empty string "". Do not add markdown blocks.
    """
    extraction = llm.invoke([SystemMessage(content=extract_sys), HumanMessage(content=latest_msg)])
    
    try:
        cleaned_json = extraction.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned_json)
    except:
        data = {"name": "", "email": "", "platform": ""}
        
    current_name = data.get("name") or state.get("user_name", "")
    current_email = data.get("email") or state.get("user_email", "")
    current_platform = data.get("platform") or state.get("user_platform", "")
    
    missing = []
    if not current_name: missing.append("Name")
    if not current_email: missing.append("Email")
    if not current_platform: missing.append("Creator Platform (like YouTube or Twitch)")
    
    if missing:
        bot_reply = f"I'd love to get you set up with the Pro plan! To get started, could I please get your {missing[0]}?"
    else:
        bot_reply = mock_lead_capture(current_name, current_email, current_platform)
        bot_reply += " I have forwarded your details to our sales team. They will be in touch shortly!"
        
    return {
        "messages": [bot_reply],
        "user_name": current_name,
        "user_email": current_email,
        "user_platform": current_platform
    }

# ==========================================
# 5. Routing Logic
# ==========================================
def route_by_intent(state: AgentState):
    """Routes the graph based on the detected intent."""
    intent = state.get("intent")
    if intent == "greeting":
        return "handle_greeting"
    elif intent == "inquiry":
        return "handle_inquiry"
    elif intent == "high_intent":
        return "process_high_intent"
    return "handle_greeting" 

# ==========================================
# 6. Graph Assembly
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("detect_intent", detect_intent)
workflow.add_node("handle_greeting", handle_greeting)
workflow.add_node("handle_inquiry", handle_inquiry)
workflow.add_node("process_high_intent", process_high_intent)

workflow.add_edge(START, "detect_intent")
workflow.add_conditional_edges(
    "detect_intent",
    route_by_intent,
    {
        "handle_greeting": "handle_greeting",
        "handle_inquiry": "handle_inquiry",
        "process_high_intent": "process_high_intent"
    }
)

workflow.add_edge("handle_greeting", END)
workflow.add_edge("handle_inquiry", END)
workflow.add_edge("process_high_intent", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)