"""Request/response models for the HTTP layer."""
from pydantic import BaseModel


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


class OrderStatusResponse(BaseModel):
    order_id: int
    status: str
    items: list[dict]


class CheckoutRequest(BaseModel):
    session_id: str


class CheckoutResponse(BaseModel):
    order_id: int
    status: str
    items: list[dict]
    created: bool  # False if this returned an existing order (idempotent replay)


class RefundDecisionResponse(BaseModel):
    order_id: int
    status: str
