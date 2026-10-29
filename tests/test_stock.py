"""
Stock is decremented when an order is placed, and an order that can't be
satisfied is refused outright (G8).
"""
import pytest
from langchain_core.messages import HumanMessage

from app.domain.repository import OutOfStockError
from app.graph import nodes


def test_checkout_decrements_stock(fake_repository):
    before = fake_repository.get_menu_item("pizza")["stock"]
    fake_repository.add_cart_item("s1", "pizza", 3)

    fake_repository.create_order_from_cart("s1")

    assert fake_repository.get_menu_item("pizza")["stock"] == before - 3


def test_checkout_refuses_when_stock_is_short(fake_repository):
    available = fake_repository.get_menu_item("pizza")["stock"]
    fake_repository.add_cart_item("s1", "pizza", available + 1)

    with pytest.raises(OutOfStockError) as exc_info:
        fake_repository.create_order_from_cart("s1")

    shortfalls = exc_info.value.shortfalls
    assert len(shortfalls) == 1
    assert shortfalls[0]["item"] == "pizza"
    assert shortfalls[0]["wanted"] == available + 1
    assert shortfalls[0]["available"] == available


def test_refused_order_leaves_stock_untouched(fake_repository):
    available = fake_repository.get_menu_item("pizza")["stock"]
    fake_repository.add_cart_item("s1", "pizza", available + 1)

    with pytest.raises(OutOfStockError):
        fake_repository.create_order_from_cart("s1")

    assert fake_repository.get_menu_item("pizza")["stock"] == available


def test_last_item_can_be_ordered_but_not_twice(fake_repository):
    """The oversell case: once the last unit is taken, the next order
    for it is refused rather than driving stock negative."""
    available = fake_repository.get_menu_item("pizza")["stock"]
    fake_repository.add_cart_item("s1", "pizza", available)
    fake_repository.create_order_from_cart("s1")
    assert fake_repository.get_menu_item("pizza")["stock"] == 0

    fake_repository.add_cart_item("s2", "pizza", 1)
    with pytest.raises(OutOfStockError):
        fake_repository.create_order_from_cart("s2")

    assert fake_repository.get_menu_item("pizza")["stock"] == 0


def test_checkout_node_reports_out_of_stock_without_ordering(
    fake_llm, fake_repository, base_state
):
    """The node must turn OutOfStockError into a normal reply, not let it
    escape as a 500, and must not set an order_id."""
    available = fake_repository.get_menu_item("pizza")["stock"]
    fake_repository.add_cart_item("test-session", "pizza", available + 1)
    state = {**base_state, "messages": [HumanMessage(content="checkout")]}

    result = nodes.checkout(state)

    assert result["order_id"] is None
    assert result["messages"]
    assert fake_repository.get_menu_item("pizza")["stock"] == available
