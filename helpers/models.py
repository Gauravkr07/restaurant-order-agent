from typing import List

from langgraph.graph import MessagesState
from pydantic import BaseModel


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
    query_count: int = 0
    cart: List[Cart]
    order: Order | None = None
    payment: dict[str, any]
