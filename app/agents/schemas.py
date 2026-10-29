"""
Structured outputs the LLM is asked to produce, and the typed results
agents hand back to the graph.
"""
from pydantic import BaseModel, Field

# A single order line can't be <= 0 (meaningless/negative) or absurdly large
# (guards against a flaky LLM extraction hallucinating a huge number).
MAX_ITEM_QUANTITY = 50


class RequestedItem(BaseModel):
    item: str
    quantity: int = Field(gt=0, le=MAX_ITEM_QUANTITY)


class UnavailableItem(BaseModel):
    item: str
    reason: str


class UnavailableItems(BaseModel):
    unavailable: list[UnavailableItem]


class AddItemVerdict(BaseModel):
    added: list[RequestedItem]
    unavailable: list[UnavailableItem]
