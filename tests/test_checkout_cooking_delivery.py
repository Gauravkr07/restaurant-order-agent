from langchain_core.messages import HumanMessage

import agent
from helpers.models import Order


def _state(base_state, cart=None, order=None, cook_retry_count=0):
    return {
        **base_state,
        "messages": [HumanMessage(content="checkout")],
        "cart": cart if cart is not None else [],
        "order": order,
        "cook_retry_count": cook_retry_count,
    }


def test_checkout_with_items_creates_order(fake_llm, base_state, sample_cart):
    result = agent.checkout(_state(base_state, cart=sample_cart))
    assert result["status"] == "ordered"
    order = result["order"]
    assert isinstance(order, Order)
    assert order.items == sample_cart
    assert order.order_id is not None


def test_checkout_empty_cart_does_not_create_order(fake_llm, base_state):
    result = agent.checkout(_state(base_state, cart=[]))
    assert "order" not in result
    assert result["messages"]


def test_cooking_stage_success_path(fake_llm, base_state, sample_cart, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.9)  # >= 0.2 -> not "failed"
    order = Order(status="ordered", order_id=1, items=sample_cart)
    result = agent.cooking_stage(_state(base_state, order=order))
    assert result["status"] == "cooking"
    assert result["cooking_duration"] == agent.COOKING_DURATION_MINUTES
    assert result["cook_retry_count"] == 0


def test_cooking_stage_failure_retries_when_attempts_remain(fake_llm, base_state, sample_cart, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.0)  # < 0.2 -> "failed"
    order = Order(status="ordered", order_id=1, items=sample_cart)
    result = agent.cooking_stage(_state(base_state, order=order, cook_retry_count=0))
    assert result["status"] == "cooking_retry"
    assert result["cook_retry_count"] == 1
    assert result["cooking_duration"] is None


def test_cooking_stage_failure_regrets_after_retries_exhausted(fake_llm, base_state, sample_cart, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.0)
    order = Order(status="ordered", order_id=1, items=sample_cart)
    result = agent.cooking_stage(
        _state(base_state, order=order, cook_retry_count=agent.MAX_COOKING_RETRIES)
    )
    assert result["status"] == "regretted"
    assert result["cook_retry_count"] == 0
    assert result["messages"]


def test_delivery_marks_order_delivered(fake_llm, base_state, sample_cart):
    order = Order(status="ordered", order_id=1, items=sample_cart)
    result = agent.delivery(_state(base_state, order=order))
    assert result["status"] == "delivered"
    assert result["order"].status == "delivered"
    assert result["order"].order_id == 1
