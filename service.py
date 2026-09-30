"""
Restaurant Agent Service
-------------------------
FastAPI layer wrapping agent.py. Redis holds only per-turn session
counters (status, query_count, cook_retry_count, unclear_count,
turn_count) - cart/order data lives in Postgres via db/repository.py,
which is the actual source of truth (see README "Data layer").

Endpoints:
  GET  /            - simple browser chat UI (static/index.html)
  GET  /menu        - list the menu
  POST /chat        - send a message (+ session_id after the first call)
"""
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import groq
import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

import agent
from db import repository
from helpers.logging_config import current_session_id, get_logger

logger = get_logger(__name__)

SESSION_TTL_SECONDS = 60 * 60 * 24  # sessions expire after 24h of inactivity
MAX_TURNS_PER_SESSION = 50  # hard cap so a session can't run forever
redis_client = redis.Redis.from_url(
    os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)

app = FastAPI(title="Restaurant Agent Service")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(groq.AuthenticationError)
def handle_groq_auth_error(request: Request, exc: groq.AuthenticationError) -> JSONResponse:
    logger.error("groq_auth_error", exc_info=True)
    return JSONResponse(
        status_code=502,
        content={"detail": "The restaurant assistant is misconfigured (invalid or expired GROQ_API_KEY). Please contact support."},
    )


@app.exception_handler(groq.RateLimitError)
def handle_groq_rate_limit(request: Request, exc: groq.RateLimitError) -> JSONResponse:
    logger.warning("groq_rate_limited")
    return JSONResponse(
        status_code=503,
        content={"detail": "The restaurant assistant is receiving too many requests right now. Please try again shortly."},
    )


@app.exception_handler(groq.APIStatusError)
def handle_groq_api_error(request: Request, exc: groq.APIStatusError) -> JSONResponse:
    logger.error("groq_api_error", exc_info=True)
    return JSONResponse(
        status_code=502,
        content={"detail": "The restaurant assistant is temporarily unavailable. Please try again shortly."},
    )


@app.exception_handler(groq.GroqError)
def handle_groq_error(request: Request, exc: groq.GroqError) -> JSONResponse:
    logger.error("groq_error", exc_info=True)
    return JSONResponse(
        status_code=502,
        content={"detail": "The restaurant assistant hit a hiccup understanding that message. Please try rephrasing or try again."},
    )


@app.exception_handler(ValidationError)
def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
    logger.warning("validation_error", extra={"errors": exc.errors()})
    return JSONResponse(
        status_code=422,
        content={"detail": "Couldn't understand the quantity or item in that message — try rephrasing (e.g. \"2 pizzas\")."},
    )


# --- session counters, keyed by session_id, persisted in Redis ---

def _session_key(session_id: str) -> str:
    return f"session:{session_id}"


def _new_session_counters() -> dict[str, Any]:
    return {
        "status": "browsing",
        "query_count": 0,
        "cook_retry_count": 0,
        "unclear_count": 0,
        "turn_count": 0,
        "order_id": None,
    }


def _load_session_counters(session_id: str) -> dict[str, Any] | None:
    raw = redis_client.get(_session_key(session_id))
    return json.loads(raw) if raw is not None else None


def _save_session_counters(session_id: str, counters: dict[str, Any]) -> None:
    redis_client.set(_session_key(session_id), json.dumps(counters), ex=SESSION_TTL_SECONDS)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "example": {"message": "show me the menu", "session_id": None}
        }
    }


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    status: str
    cart: list[dict]
    order: dict | None = None
    order_id: int | None = None
    cooking_duration: int | None = None


@app.get("/menu")
def get_menu() -> list[dict]:
    return repository.get_active_menu_items()


@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    start_time = time.monotonic()
    if req.session_id is None:
        session_id = str(uuid.uuid4())
        counters = _new_session_counters()
        repository.get_or_create_session(session_id)
        current_session_id.set(session_id)
        logger.info("session_created")
    else:
        session_id = req.session_id
        current_session_id.set(session_id)
        counters = _load_session_counters(session_id)
        if counters is None:
            logger.warning("unknown_session_id")
            raise HTTPException(status_code=404, detail="Unknown session_id")
        if counters["status"] == "regretted":
            logger.info("rejected_closed_session")
            raise HTTPException(
                status_code=409,
                detail="This conversation has ended. Please start a new session to order again.",
            )
        if counters["turn_count"] >= MAX_TURNS_PER_SESSION:
            logger.warning("session_turn_limit_reached", extra={"turn_count": counters["turn_count"]})
            raise HTTPException(
                status_code=409,
                detail="This conversation has gone on for a while - please start a new session to continue.",
            )

    counters["turn_count"] += 1
    logger.info("chat_request", extra={"turn_count": counters["turn_count"], "message_length": len(req.message)})

    state = {
        "messages": [HumanMessage(content=req.message)],
        "session_id": session_id,
        "status": counters["status"],
        "query_count": counters["query_count"],
        "cook_retry_count": counters["cook_retry_count"],
        "unclear_count": counters["unclear_count"],
        "order_id": counters["order_id"],
    }
    try:
        result = agent.graph.invoke(state)
    except groq.GroqError:
        # Turn never completed - don't persist counters, just give a
        # graceful reply so the customer can retry from where they were.
        logger.error(
            "llm_failure_fallback",
            extra={"duration_ms": round((time.monotonic() - start_time) * 1000)},
            exc_info=True,
        )
        cart = repository.get_cart_items(session_id)
        order = repository.get_order(counters["order_id"]) if counters["order_id"] else None
        return ChatResponse(
            session_id=session_id,
            reply="Sorry, I'm having trouble processing that right now. Please try again in a moment.",
            status=counters["status"],
            cart=cart,
            order=order,
            order_id=counters["order_id"],
            cooking_duration=None,
        )

    counters = {
        "status": result.get("status", counters["status"]),
        "query_count": result.get("query_count", counters["query_count"]),
        "cook_retry_count": result.get("cook_retry_count", counters["cook_retry_count"]),
        "unclear_count": result.get("unclear_count", counters["unclear_count"]),
        "turn_count": counters["turn_count"],
        "order_id": result.get("order_id", counters["order_id"]),
    }
    _save_session_counters(session_id, counters)

    duration_ms = round((time.monotonic() - start_time) * 1000)
    logger.info("chat_response", extra={"status": counters["status"], "duration_ms": duration_ms})

    last_message = result["messages"][-1]
    content = getattr(last_message, "content", last_message)
    cart = repository.get_cart_items(session_id)
    order = repository.get_order(counters["order_id"]) if counters["order_id"] else None
    return ChatResponse(
        session_id=session_id,
        reply=content,
        status=counters["status"],
        cart=cart,
        order=order,
        order_id=counters["order_id"],
        cooking_duration=result.get("cooking_duration"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
