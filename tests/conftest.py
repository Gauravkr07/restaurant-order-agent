import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from helpers.models import Cart, RequestedItem, RequestedItems


class FakeStructuredExtractor:
    """Stands in for llm.with_structured_output(RequestedItems).

    `result` is either a RequestedItems instance to return, or an exception
    (or list of exceptions/results, consumed in order) to raise - lets tests
    simulate the Groq flakiness that extract_requested_items() retries on.
    """

    def __init__(self, result):
        self._queue = result if isinstance(result, list) else [result]

    def invoke(self, messages):
        outcome = self._queue.pop(0) if len(self._queue) > 1 else self._queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeLLM:
    """Stands in for helpers.llm.llm.

    - `intent`: what route_intent's classification call should return.
    - `reply_text`: content for any plain llm.invoke() call (confirmations,
      menu text, apologies, etc.) - tests should not assert on this wording.
    - `extraction_result`: what with_structured_output(...).invoke() returns;
      see FakeStructuredExtractor for the retry-simulation shape.
    """

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


@pytest.fixture
def fake_llm(monkeypatch):
    llm = FakeLLM()
    monkeypatch.setattr("agent.llm", llm)
    return llm


@pytest.fixture
def sample_cart():
    return [
        Cart(item="pizza", quantity=2, price=200.0),
        Cart(item="coke", quantity=1, price=50.0),
    ]


@pytest.fixture
def base_state():
    return {
        "messages": [],
        "status": "browsing",
        "query_count": 0,
        "cook_retry_count": 0,
        "unclear_count": 0,
        "cooking_duration": None,
        "cart": [],
        "order": None,
    }
