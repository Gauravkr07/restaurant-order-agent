"""
The conversational endpoint: one customer message in, one agent reply out.

Session counters are loaded from and saved back to Redis around each
graph run (see app/domain/session_store.py); cart/order data is read
straight from Postgres, which is the source of truth.
"""
import time
import uuid

import groq
from fastapi import APIRouter, HTTPException
from langchain_core.messages import HumanMessage

from app.api.schemas import ChatRequest, ChatResponse
from app.core.config import settings
from app.core.logging import current_session_id, get_logger
from app.domain import repository, session_store
from app.graph import graph
from app.graph.state import Status

logger = get_logger(__name__)
router = APIRouter(tags=["chat"])


def _start_session() -> tuple[str, dict]:
    session_id = str(uuid.uuid4())
    repository.get_or_create_session(session_id)
    current_session_id.set(session_id)
    logger.info("session_created")
    return session_id, session_store.new_counters()


def _resume_session(session_id: str) -> dict:
    current_session_id.set(session_id)
    counters = session_store.load(session_id)
    if counters is None:
        logger.warning("unknown_session_id")
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if counters["status"] == Status.REGRETTED:
        logger.info("rejected_closed_session")
        raise HTTPException(
            status_code=409,
            detail="This conversation has ended. Please start a new session to order again.",
        )
    if counters["turn_count"] >= settings.max_turns_per_session:
        logger.warning(
            "session_turn_limit_reached", extra={"turn_count": counters["turn_count"]}
        )
        raise HTTPException(
            status_code=409,
            detail="This conversation has gone on for a while - please start a "
            "new session to continue.",
        )
    return counters


def _build_response(session_id: str, reply: str, counters: dict, cooking_duration=None) -> ChatResponse:
    order_id = counters["order_id"]
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        status=counters["status"],
        cart=repository.get_cart_items(session_id),
        order=repository.get_order(order_id) if order_id else None,
        order_id=order_id,
        cooking_duration=cooking_duration,
    )


@router.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    start_time = time.monotonic()

    if req.session_id is None:
        session_id, counters = _start_session()
    else:
        session_id = req.session_id
        counters = _resume_session(session_id)

    counters["turn_count"] += 1
    logger.info(
        "chat_request",
        extra={"turn_count": counters["turn_count"], "message_length": len(req.message)},
    )

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
        result = graph.invoke(state)
    except groq.GroqError:
        # Turn never completed - don't persist counters, just give a
        # graceful reply so the customer can retry from where they were.
        logger.error(
            "llm_failure_fallback",
            extra={"duration_ms": round((time.monotonic() - start_time) * 1000)},
            exc_info=True,
        )
        return _build_response(
            session_id,
            "Sorry, I'm having trouble processing that right now. Please try again in a moment.",
            counters,
        )

    counters = session_store.merge_result(counters, result)
    session_store.save(session_id, counters)

    logger.info(
        "chat_response",
        extra={
            "status": counters["status"],
            "duration_ms": round((time.monotonic() - start_time) * 1000),
        },
    )

    last_message = result["messages"][-1]
    reply = getattr(last_message, "content", last_message)
    return _build_response(session_id, reply, counters, result.get("cooking_duration"))
