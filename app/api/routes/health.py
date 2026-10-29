"""
Health checks for the deployment platform.

Two separate things, deliberately:

- /healthz answers "is this process up?" and touches nothing external.
  Render polls it to decide whether the instance is live, so it must not
  fail because a dependency is briefly unavailable - that would take the
  whole service out of rotation over a transient blip.

- /readyz answers "can this process actually serve traffic?" and checks
  Postgres and Redis. Use it after a deploy, or when diagnosing.
"""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.logging import get_logger
from app.domain.base import engine
from app.domain.session_store import redis_client

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return {"status": "ok"}


@router.get("/readyz", include_in_schema=False)
def readyz() -> JSONResponse:
    checks: dict[str, str] = {}

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        logger.warning("readyz_postgres_failed", exc_info=True)
        checks["postgres"] = f"error: {type(exc).__name__}"

    try:
        redis_client.ping()
        checks["redis"] = "ok"
    except Exception as exc:
        logger.warning("readyz_redis_failed", exc_info=True)
        checks["redis"] = f"error: {type(exc).__name__}"

    ready = all(value == "ok" for value in checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"ready": ready, "checks": checks},
    )
