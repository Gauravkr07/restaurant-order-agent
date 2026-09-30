from langchain_core.messages import HumanMessage

import agent
from helpers.models import UnavailableItems


def _state(base_state, message):
    return {**base_state, "messages": [HumanMessage(content=message)]}


def test_adding_new_item_creates_cart_line(fake_llm, fake_repository, base_state):
    fake_llm.tool_calls = [
        {"name": "get_menu_item", "args": {"name": "burger"}, "id": "1"},
        {"name": "add_to_cart", "args": {"session_id": "test-session", "item_name": "burger", "quantity": 1}, "id": "2"},
    ]
    fake_llm.verdict = UnavailableItems(unavailable=[])
    result = agent.verify_items(_state(base_state, "1 burger"))
    assert fake_repository.get_cart_items("test-session") == [{"item": "burger", "quantity": 1, "price": 150.0}]
    assert result["status"] == "cart"


def test_adding_more_of_existing_item_sums_quantity_not_replaces(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 1)
    fake_llm.tool_calls = [
        {"name": "add_to_cart", "args": {"session_id": "test-session", "item_name": "pizza", "quantity": 2}, "id": "1"},
    ]
    fake_llm.verdict = UnavailableItems(unavailable=[])
    agent.verify_items(_state(base_state, "2 more pizzas"))
    assert fake_repository.get_cart_item_quantity("test-session", "pizza") == 3


def test_quantity_exceeding_stock_is_skipped(fake_llm, fake_repository, base_state):
    # The tool-based agent is instructed not to call add_to_cart when
    # stock is insufficient - it just reports the item as unavailable.
    fake_llm.tool_calls = [
        {"name": "get_menu_item", "args": {"name": "pizza"}, "id": "1"},
    ]
    fake_llm.verdict = UnavailableItems(
        unavailable=[{"item": "pizza", "reason": "only 20 in stock, asked for 30"}]
    )
    agent.verify_items(_state(base_state, "30 pizzas"))
    assert fake_repository.get_cart_items("test-session") == []
