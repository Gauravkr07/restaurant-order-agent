"""
Structured JSON logging, configured once and shared across the app.

Usage: `from helpers.logging_config import get_logger` then
`logger = get_logger(__name__)`, then `logger.info("did_thing", extra={...})`.

Logs go to stdout only - Docker/Compose already captures and retains
stdout (`docker compose logs`), so there's no file to manage or rotate.
Each line is a single JSON object, which keeps them both human-readable
in a terminal and trivially parseable by any log aggregator later.
"""
import contextvars
import json
import logging
import os
import sys

_LOG_RECORD_BUILTIN_FIELDS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)

# Set once per /chat request (see service.py) so every log line emitted
# while handling that request - including from agent.py's node functions,
# which have no session_id parameter of their own - carries the same
# session_id without having to thread it through every function call.
current_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_session_id", default=None
)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        session_id = current_session_id.get()
        if session_id is not None:
            payload["session_id"] = session_id
        # Anything passed via logger.info("msg", extra={...}) shows up as
        # extra attributes on the record - pull those in as structured
        # fields alongside the message, e.g. session_id, node, duration_ms.
        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_BUILTIN_FIELDS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _configure_root_logger() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured (e.g. re-imported under --reload)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())


def get_logger(name: str) -> logging.Logger:
    _configure_root_logger()
    return logging.getLogger(name)
