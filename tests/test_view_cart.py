from langchain_core.messages import HumanMessage

import agent


def _state(base_state):
    return {**base_state, "messages": [HumanMessage(content="show my cart")]}


def test_view_cart_with_items_returns_a_reply(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 2)
    result = agent.view_cart(_state(base_state))
    assert result["messages"]
    assert result["unclear_count"] == 0


def test_view_cart_empty_does_not_crash_on_total(fake_llm, fake_repository, base_state):
    result = agent.view_cart(_state(base_state))
    assert result["messages"]
    assert result["unclear_count"] == 0
