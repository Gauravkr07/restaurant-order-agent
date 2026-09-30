import groq
import pytest

import agent
from helpers.models import RequestedItem, RequestedItems


def test_retries_once_after_a_groq_error_then_succeeds(fake_llm):
    success = RequestedItems(items=[RequestedItem(item="pizza", quantity=1)])
    fake_llm.extraction_result = [groq.APIError("flaked", request=None, body=None), success]

    result = agent.extract_requested_items("2 pizzas", "system prompt")

    assert result == success


def test_raises_after_exhausting_retry_attempts(fake_llm):
    error = groq.APIError("still broken", request=None, body=None)
    fake_llm.extraction_result = [error, error]

    with pytest.raises(groq.GroqError):
        agent.extract_requested_items("2 pizzas", "system prompt")
