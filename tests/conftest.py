import os
import sys
from pathlib import Path

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers.models import RequestedItem, RequestedItems

MENU = {
    "pizza": {"price": 200.0, "stock": 20},
    "burger": {"price": 150.0, "stock": 15},
    "coke": {"price": 50.0, "stock": 50},
}


class FakeStructuredExtractor:
    def __init__(self, result):
        self._queue = result if isinstance(result, list) else [result]

    def invoke(self, messages):
        outcome = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeLLM:
    def __init__(self, intent="SHOW_MENU", reply_text="ok", extraction_result=None):
        self.intent = intent
        self.reply_text = reply_text
        self.extraction_result = (
            extraction_result if extraction_result is not None else RequestedItems(items=[])
        )
        self.classify_calls = 0
        self.invoke_calls = 0

    def invoke(self, messages):
        self.invoke_calls += 1
        system_text = messages[0][1] if messages else ""
        if "Classify" in system_text:
            self.classify_calls += 1
            return AIMessage(content=self.intent)
        return AIMessage(content=self.reply_text)

    def with_structured_output(self, schema):
        return FakeStructuredExtractor(self.extraction_result)


class FakeRepository:
    """In-memory stand-in for db.repository, keyed by session_id, so the
    unit test suite stays offline/fast - no real Postgres needed here."""

    def __init__(self):
        self.carts: dict[str, dict[str, int]] = {}
        self.orders: dict[int, dict] = {}
        self._next_order_id = 1

    def get_active_menu_items(self):
        return [{"name": k, **v} for k, v in MENU.items()]

    def get_menu_item(self, name):
        return {"name": name, **MENU[name]} if name in MENU else None

    def get_cart_items(self, session_id):
        cart = self.carts.get(session_id, {})
        return [{"item": k, "quantity": q, "price": MENU[k]["price"]} for k, q in cart.items()]

    def add_cart_item(self, session_id, item_name, quantity):
        cart = self.carts.setdefault(session_id, {})
        cart[item_name] = cart.get(item_name, 0) + quantity

    def get_cart_item_quantity(self, session_id, item_name):
        return self.carts.get(session_id, {}).get(item_name, 0)

    def remove_cart_item(self, session_id, item_name, quantity):
        cart = self.carts.get(session_id, {})
        if item_name not in cart:
            return "not_in_cart"
        remaining = cart[item_name] - quantity
        if remaining < 0:
            return "insufficient"
        if remaining == 0:
            del cart[item_name]
            return "removed"
        cart[item_name] = remaining
        return "updated"

    def create_order_from_cart(self, session_id):
        cart = self.carts.get(session_id, {})
        if not cart:
            return None
        items = [{"item": k, "quantity": q, "price": MENU[k]["price"]} for k, q in cart.items()]
        order_id = self._next_order_id
        self._next_order_id += 1
        order = {"order_id": order_id, "status": "ordered", "items": items}
        self.orders[order_id] = order
        self.carts[session_id] = {}
        return order

    def get_order(self, order_id):
        return self.orders.get(order_id)

    def update_order_status(self, order_id, status):
        order = self.orders.get(order_id)
        if order is None:
            return None
        order["status"] = status
        return order


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("agent.llm", llm)
    return llm


@pytest.fixture
def fake_repository(monkeypatch):
    repo = FakeRepository()
    monkeypatch.setattr("agent.repository", repo)
    return repo


@pytest.fixture
def base_state():
    return {
        "messages": [],
        "session_id": "test-session",
        "status": "browsing",
        "query_count": 0,
        "cook_retry_count": 0,
        "unclear_count": 0,
        "cooking_duration": None,
        "order_id": None,
    }
