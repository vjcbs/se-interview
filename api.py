import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# --- Phoenix tracing: must run before the agent (and LangChain) is created ---
from openinference.instrumentation import using_session
from phoenix.otel import register

register(
    project_name="arize-travel-agent",  # groups all traces under one Phoenix project
    auto_instrument=True,  # auto-instruments LangChain/LangGraph: LLM calls, tools, graph nodes
    batch=True,  # export spans in a background thread so tracing never slows down responses
)

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent import build_agent
from policy_tool import load_policy

STATIC_DIR = Path(__file__).parent / "static"

agent = build_agent()

app = FastAPI(
    title="Arize Corp Travel Assistant API",
    description="A policy-aware LangGraph travel agent: flights, hotels and attractions within company policy",
    version="0.2.0",
)


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # reuse to continue a conversation


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the agent. Pass the returned session_id to continue the conversation."""
    session_id = request.session_id or str(uuid.uuid4())
    # using_session tags every span with session.id so Phoenix groups multi-turn conversations
    with using_session(session_id):
        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config={"configurable": {"thread_id": session_id}},
        )
    return ChatResponse(response=result["messages"][-1].content, session_id=session_id)


@app.get("/", include_in_schema=False)
def ui():
    """Serve the chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/policy")
def policy():
    """Current company travel policy (shown in the UI's policy panel)."""
    return load_policy()


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}
