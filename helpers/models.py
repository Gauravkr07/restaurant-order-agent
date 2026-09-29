from typing import Any, List, Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel

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


class RequestedItem(BaseModel):
    item: str
    quantity: int


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
    cooking_duration: int | None = None
    cart: List[Cart]
    order: Order | None = None
    payment: dict[str, Any] = {}
