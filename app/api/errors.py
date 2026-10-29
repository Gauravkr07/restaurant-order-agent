"""
Every exception handler in one place.

Each handler logs the real cause server-side and returns a short,
customer-safe message - a traceback or driver error string never reaches
the client.
"""
import groq
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.logging import get_logger

logger = get_logger(__name__)


def _json(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def handle_groq_auth_error(request: Request, exc: groq.AuthenticationError) -> JSONResponse:
    logger.error("groq_auth_error", exc_info=True)
    return _json(
        502,
        "The restaurant assistant is misconfigured (invalid or expired "
        "GROQ_API_KEY). Please contact support.",
    )


def handle_groq_rate_limit(request: Request, exc: groq.RateLimitError) -> JSONResponse:
    logger.warning("groq_rate_limited")
    return _json(
        503,
        "The restaurant assistant is receiving too many requests right now. "
        "Please try again shortly.",
    )


def handle_groq_api_error(request: Request, exc: groq.APIStatusError) -> JSONResponse:
    logger.error("groq_api_error", exc_info=True)
    return _json(
        502,
        "The restaurant assistant is temporarily unavailable. Please try again shortly.",
    )


def handle_groq_error(request: Request, exc: groq.GroqError) -> JSONResponse:
    logger.error("groq_error", exc_info=True)
    return _json(
        502,
        "The restaurant assistant hit a hiccup understanding that message. "
        "Please try rephrasing or try again.",
    )


def handle_validation_error(request: Request, exc: ValidationError) -> JSONResponse:
    logger.warning("validation_error", extra={"errors": exc.errors()})
    return _json(
        422,
        'Couldn\'t understand the quantity or item in that message — try '
        'rephrasing (e.g. "2 pizzas").',
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Handlers are registered most-specific first; FastAPI matches on
    exception class, so the GroqError catch-all must not shadow the
    specific Groq subclasses above it."""
    app.add_exception_handler(groq.AuthenticationError, handle_groq_auth_error)
    app.add_exception_handler(groq.RateLimitError, handle_groq_rate_limit)
    app.add_exception_handler(groq.APIStatusError, handle_groq_api_error)
    app.add_exception_handler(groq.GroqError, handle_groq_error)
    app.add_exception_handler(ValidationError, handle_validation_error)
