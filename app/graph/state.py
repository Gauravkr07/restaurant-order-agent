"""
Conversation state carried through the graph, plus the status enum that
constrains it.

Status was previously a bare typing.Literal that nothing enforced at
runtime. As a str-Enum it stays compatible with the plain strings used in
Redis/JSON and in comparisons (Status.CART == "cart" is True), while
giving one authoritative list of valid statuses.
"""
from enum import StrEnum

from langgraph.graph import MessagesState


class Status(StrEnum):
    BROWSING = "browsing"
    VERIFYING = "verifying"
    CART = "cart"
    ORDERED = "ordered"
    COOKING = "cooking"
    COOKING_RETRY = "cooking_retry"
    DELIVERED = "delivered"
    REGRETTED = "regretted"


class State(MessagesState):
    session_id: str
    status: str = Status.BROWSING
    query_count: int = 0
    cook_retry_count: int = 0
    unclear_count: int = 0
    cooking_duration: int | None = None
    order_id: int | None = None
