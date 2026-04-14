# AutoStream AI Agent

A conversational AI agent for AutoStream — a fictional SaaS company offering automated video editing tools. Built as part of the ServiceHive / Inflx ML Intern assignment.

---

## How to Run Locally

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd autostream
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.env` file in the project root:
```
OPENROUTER_API_KEY=your_openrouter_key_here
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Architecture (~200 words)

This agent is built using **LangGraph**, a graph-based orchestration framework built on LangChain. LangGraph was chosen over alternatives like AutoGen because it provides explicit, inspectable control flow — each node in the graph has a single responsibility, making the agent's reasoning transparent and easy to debug.

**State Management:** LangGraph's `StateGraph` holds a typed `AgentState` dictionary that persists across conversation turns via `MemorySaver` (an in-memory checkpointer). This state tracks the full message history plus extracted lead fields (`user_name`, `user_email`, `user_platform`) and the current `intent`. Because state is passed through every node, the agent can accumulate user details across multiple turns without losing context.

**Flow:** Every user message enters the `detect_intent` node first, which classifies it into `greeting`, `inquiry`, or `high_intent`. A conditional router then directs execution to the correct handler. `handle_inquiry` uses a local ChromaDB vector store (embedded with `all-MiniLM-L6-v2`) to retrieve relevant chunks from the knowledge base and answer questions accurately. `process_high_intent` extracts lead fields turn-by-turn and only fires `mock_lead_capture()` once all three fields are collected.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp:

1. **Register a WhatsApp Business API account** via Meta for Developers and obtain a phone number ID and access token.
2. **Set up a webhook endpoint** (e.g. FastAPI or Flask) that Meta will POST incoming messages to. Verify the webhook using the `hub.verify_token` challenge.
3. **Route incoming messages** to the LangGraph agent — pass the sender's WhatsApp ID as the `thread_id` so each user gets isolated, persistent memory.
4. **Reply using the WhatsApp API**: after the agent returns a response, POST it back to `https://graph.facebook.com/v17.0/{phone_number_id}/messages` with the recipient's number and message text.
5. **Deploy** the webhook server on a public HTTPS URL (e.g. Railway, Render, or ngrok for local testing).

This approach keeps the agent logic completely decoupled from the messaging layer — the same LangGraph agent can serve WhatsApp, Telegram, or a web UI by simply swapping the I/O layer.