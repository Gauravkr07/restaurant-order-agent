"""
Per-turn session counters in Redis, keyed by session_id.

Only counters live here (status, query_count, cook_retry_count,
unclear_count, turn_count, order_id) - cart and order data live in
Postgres via app/domain/repository.py, which is the source of truth.

Extracted from service.py so the HTTP layer no longer talks to Redis
directly and the storage choice stays swappable.
"""
import json
from typing import Any

import redis

from app.core.config import settings
from app.graph.state import Status

redis_client = redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _key(session_id: str) -> str:
    return f"session:{session_id}"


def new_counters() -> dict[str, Any]:
    return {
        "status": str(Status.BROWSING),
        "query_count": 0,
        "cook_retry_count": 0,
        "unclear_count": 0,
        "turn_count": 0,
        "order_id": None,
    }


def load(session_id: str) -> dict[str, Any] | None:
    raw = redis_client.get(_key(session_id))
    return json.loads(raw) if raw is not None else None


def save(session_id: str, counters: dict[str, Any]) -> None:
    redis_client.set(
        _key(session_id), json.dumps(counters), ex=settings.session_ttl_seconds
    )


def merge_result(counters: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Fold a graph run's state update back into the session counters,
    keeping the previous value for anything the run didn't set."""
    return {
        "status": str(result.get("status", counters["status"])),
        "query_count": result.get("query_count", counters["query_count"]),
        "cook_retry_count": result.get("cook_retry_count", counters["cook_retry_count"]),
        "unclear_count": result.get("unclear_count", counters["unclear_count"]),
        "turn_count": counters["turn_count"],
        "order_id": result.get("order_id", counters["order_id"]),
    }
