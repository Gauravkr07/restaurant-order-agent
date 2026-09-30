from typing import Any, List, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

Status = Literal[
    "browsing",
    "verifying",
    "cart",
    "ordered",
    "cooking",
    "cooking_retry",
    "delivered",
    "regretted",
]

# A single order line can't be <= 0 (meaningless/negative) or absurdly large
# (guards against a flaky LLM extraction hallucinating a huge number).
MAX_ITEM_QUANTITY = 50


class RequestedItem(BaseModel):
    item: str
    quantity: int = Field(gt=0, le=MAX_ITEM_QUANTITY)


class RequestedItems(BaseModel):
    items: list[RequestedItem]


class Cart(BaseModel):
    item: str
    quantity: int
    price: float


class Order(BaseModel):
    status: str
    order_id: int
    items: list[Cart]


class State(MessagesState):
    status: Status = "browsing"
    query_count: int = 0
    cook_retry_count: int = 0
    unclear_count: int = 0
    turn_count: int = 0
    cooking_duration: int | None = None
    cart: List[Cart]
    order: Order | None = None
    payment: dict[str, Any] = {}
