from langchain_core.messages import HumanMessage

import agent


def _state(base_state, order_id=None, cook_retry_count=0):
    return {
        **base_state,
        "messages": [HumanMessage(content="checkout")],
        "order_id": order_id,
        "cook_retry_count": cook_retry_count,
    }


def test_checkout_with_items_creates_order(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 2)
    result = agent.checkout(_state(base_state))
    assert result["status"] == "ordered"
    assert result["order_id"] is not None
    order = fake_repository.get_order(result["order_id"])
    assert order["items"] == [{"item": "pizza", "quantity": 2, "price": 200.0}]


def test_checkout_empty_cart_does_not_create_order(fake_llm, fake_repository, base_state):
    result = agent.checkout(_state(base_state))
    # order_id is explicitly None (not just omitted) so a stale order_id
    # from a PREVIOUS order in this session can't leak through and get
    # treated as "a new order was just placed" by route_after_checkout.
    assert result["order_id"] is None
    assert result["messages"]


def test_route_after_checkout_skips_cooking_on_empty_cart(fake_llm, fake_repository, base_state):
    result = agent.checkout(_state(base_state))
    merged_state = {**_state(base_state), **result}
    assert agent.route_after_checkout(merged_state) == "END"


def test_route_after_checkout_goes_to_cooking_on_real_order(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 1)
    result = agent.checkout(_state(base_state))
    merged_state = {**_state(base_state), **result}
    assert agent.route_after_checkout(merged_state) == "COOK"


def test_cooking_stage_success_path(fake_llm, fake_repository, base_state, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.9)  # >= 0.2 -> not "failed"
    result = agent.cooking_stage(_state(base_state, order_id=1))
    assert result["status"] == "cooking"
    assert result["cooking_duration"] == agent.COOKING_DURATION_MINUTES
    assert result["cook_retry_count"] == 0


def test_cooking_stage_failure_retries_when_attempts_remain(fake_llm, fake_repository, base_state, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.0)  # < 0.2 -> "failed"
    result = agent.cooking_stage(_state(base_state, order_id=1, cook_retry_count=0))
    assert result["status"] == "cooking_retry"
    assert result["cook_retry_count"] == 1
    assert result["cooking_duration"] is None


def test_cooking_stage_failure_regrets_after_retries_exhausted(fake_llm, fake_repository, base_state, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.0)
    fake_repository.add_cart_item("test-session", "pizza", 1)
    order = fake_repository.create_order_from_cart("test-session")
    result = agent.cooking_stage(
        _state(base_state, order_id=order["order_id"], cook_retry_count=agent.MAX_COOKING_RETRIES)
    )
    assert result["status"] == "regretted"
    assert result["cook_retry_count"] == 0
    assert result["messages"]
    assert fake_repository.get_order(order["order_id"])["status"] == "refunded"


def test_cooking_stage_large_refund_needs_human_approval(fake_llm, fake_repository, base_state, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.0)
    fake_repository.add_cart_item("test-session", "pizza", 50)  # 50 x 200 = 10,000
    fake_repository.add_cart_item("test-session", "burger", 1)  # tips it over 10,000
    order = fake_repository.create_order_from_cart("test-session")
    result = agent.cooking_stage(
        _state(base_state, order_id=order["order_id"], cook_retry_count=agent.MAX_COOKING_RETRIES)
    )
    assert result["status"] == "regretted"
    assert fake_repository.get_order(order["order_id"])["status"] == "pending_refund_approval"


def test_cooking_stage_small_refund_auto_approved(fake_llm, fake_repository, base_state, monkeypatch):
    monkeypatch.setattr(agent.random, "random", lambda: 0.0)
    fake_repository.add_cart_item("test-session", "pizza", 1)
    order = fake_repository.create_order_from_cart("test-session")
    result = agent.cooking_stage(
        _state(base_state, order_id=order["order_id"], cook_retry_count=agent.MAX_COOKING_RETRIES)
    )
    assert result["status"] == "regretted"
    assert fake_repository.get_order(order["order_id"])["status"] == "refunded"


def test_delivery_marks_order_delivered(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 2)
    order = fake_repository.create_order_from_cart("test-session")
    result = agent.delivery(_state(base_state, order_id=order["order_id"]))
    assert result["status"] == "delivered"
    assert fake_repository.get_order(order["order_id"])["status"] == "delivered"
