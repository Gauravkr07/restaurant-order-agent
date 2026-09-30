from langchain_core.messages import HumanMessage

import agent


def _state(base_state, cart):
    return {**base_state, "messages": [HumanMessage(content="show my cart")], "cart": cart}


def test_view_cart_with_items_returns_a_reply(fake_llm, base_state, sample_cart):
    result = agent.view_cart(_state(base_state, sample_cart))
    assert result["messages"]
    assert result["unclear_count"] == 0


def test_view_cart_empty_does_not_crash_on_total(fake_llm, base_state):
    result = agent.view_cart(_state(base_state, []))
    assert result["messages"]
    assert result["unclear_count"] == 0
