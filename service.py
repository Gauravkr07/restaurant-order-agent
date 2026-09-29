"""
Restaurant Agent Service
-------------------------
A thin FastAPI service that exposes the LangGraph restaurant agent
(defined in agent.py) over HTTP. agent.py is imported as-is and is
not modified by this file.

The graph is stateful across turns (status, query_count, cook_retry_count,
cart, order), but that state lives on the server, not the client. /chat
takes a session_id: on the first call omit it and one is generated and
returned; on every later call pass it back and the server resumes the
conversation from wherever it left off. The client never sees or sends
query_count/cook_retry_count/status/cart/order directly - it only ever
sends a message.

Endpoints:
  GET  /            - simple browser chat UI (static/index.html)
  GET  /menu        - list the menu
  POST /chat        - send a message (+ session_id after the first call)
"""
import uuid
from pathlib import Path
from typing import Any

import groq
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

import agent

app = FastAPI(title="Restaurant Agent Service")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(groq.AuthenticationError)
def handle_groq_auth_error(request: Request, exc: groq.AuthenticationError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"detail": "The restaurant assistant is misconfigured (invalid or expired GROQ_API_KEY). Please contact support."},
    )


@app.exception_handler(groq.RateLimitError)
def handle_groq_rate_limit(request: Request, exc: groq.RateLimitError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "The restaurant assistant is receiving too many requests right now. Please try again shortly."},
    )


@app.exception_handler(groq.APIStatusError)
def handle_groq_api_error(request: Request, exc: groq.APIStatusError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"detail": "The restaurant assistant is temporarily unavailable. Please try again shortly."},
    )


@app.exception_handler(groq.GroqError)
def handle_groq_error(request: Request, exc: groq.GroqError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={"detail": "The restaurant assistant hit a hiccup understanding that message. Please try rephrasing or try again."},
    )

# --- in-memory conversation state, keyed by session_id ---
_sessions: dict[str, dict[str, Any]] = {}


def _new_session_state() -> dict[str, Any]:
    return {
        "status": "browsing",
        "query_count": 0,
        "cook_retry_count": 0,
        "cart": [],
        "order": None,
    }


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "show me the menu",
                "session_id": None,
            }
        }
    }


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: str
    cart: list[agent.Cart]
    order: agent.Order | None = None
    order_id: int | None = None
    cooking_duration: int | None = None


@app.get("/menu")
def get_menu() -> dict[str, Any]:
    return agent.MENU


@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    if req.session_id is None:
        session_id = str(uuid.uuid4())
        session_state = _new_session_state()
    else:
        session_id = req.session_id
        session_state = _sessions.get(session_id)
        if session_state is None:
            raise HTTPException(status_code=404, detail="Unknown session_id")

    state = {
        "messages": [HumanMessage(content=req.message)],
        **session_state,
    }
    result = agent.graph.invoke(state)

    session_state = {
        "status": result.get("status", session_state["status"]),
        "query_count": result.get("query_count", session_state["query_count"]),
        "cook_retry_count": result.get("cook_retry_count", session_state["cook_retry_count"]),
        "cart": result.get("cart", session_state["cart"]),
        "order": result.get("order", session_state["order"]),
    }
    _sessions[session_id] = session_state

    last_message = result["messages"][-1]
    content = getattr(last_message, "content", last_message)
    order = session_state["order"]
    return ChatResponse(
        session_id=session_id,
        reply=content,
        status=session_state["status"],
        cart=session_state["cart"],
        order=order,
        order_id=order.order_id if order else None,
        cooking_duration=result.get("cooking_duration"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
