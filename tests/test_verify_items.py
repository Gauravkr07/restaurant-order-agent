from langchain_core.messages import HumanMessage

import agent
from helpers.models import UnavailableItems


def _state(base_state, message, query_count=0):
    return {
        **base_state,
        "messages": [HumanMessage(content=message)],
        "query_count": query_count,
    }


def test_all_items_available_moves_to_cart(fake_llm, fake_repository, base_state):
    fake_llm.tool_calls = [
        {"name": "add_to_cart", "args": {"session_id": "test-session", "item_name": "pizza", "quantity": 2}, "id": "1"},
    ]
    fake_llm.verdict = UnavailableItems(unavailable=[])
    result = agent.verify_items(_state(base_state, "2 pizzas"))
    assert result["status"] == "cart"
    assert result["query_count"] == 0
    assert result["unclear_count"] == 0


def test_partial_availability_asks_user_and_increments_query_count(fake_llm, fake_repository, base_state):
    fake_llm.verdict = UnavailableItems(
        unavailable=[{"item": "pizza", "reason": "only 20 in stock, asked for 30"}]
    )
    result = agent.verify_items(_state(base_state, "30 pizzas", query_count=0))
    assert result["status"] == "verifying"
    assert result["query_count"] == 1
    assert result["messages"]


def test_third_failed_attempt_regrets(fake_llm, fake_repository, base_state):
    fake_llm.verdict = UnavailableItems(
        unavailable=[{"item": "pizza", "reason": "only 20 in stock, asked for 30"}]
    )
    result = agent.verify_items(
        _state(base_state, "30 pizzas", query_count=agent.MAX_VERIFY_ATTEMPTS - 1)
    )
    assert result["status"] == "regretted"
    assert result["query_count"] == agent.MAX_VERIFY_ATTEMPTS
    assert result["messages"]


def test_item_not_on_menu_is_treated_as_unavailable(fake_llm, fake_repository, base_state):
    fake_llm.verdict = UnavailableItems(
        unavailable=[{"item": "lobster thermidor", "reason": "not on our menu"}]
    )
    result = agent.verify_items(_state(base_state, "1 lobster thermidor"))
    assert result["status"] == "verifying"
    assert result["query_count"] == 1
