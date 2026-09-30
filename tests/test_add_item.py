from langchain_core.messages import HumanMessage

import agent
from helpers.models import RequestedItem, RequestedItems


def _state(base_state, message):
    return {**base_state, "messages": [HumanMessage(content=message)]}


def test_adding_new_item_creates_cart_line(fake_llm, fake_repository, base_state):
    fake_llm.extraction_result = RequestedItems(items=[RequestedItem(item="burger", quantity=1)])
    result = agent.add_item(_state(base_state, "1 burger"))
    cart = fake_repository.get_cart_items("test-session")
    assert cart == [{"item": "burger", "quantity": 1, "price": 150.0}]
    assert result["status"] == "cart"


def test_adding_more_of_existing_item_sums_quantity_not_replaces(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 1)
    fake_llm.extraction_result = RequestedItems(items=[RequestedItem(item="pizza", quantity=2)])
    agent.add_item(_state(base_state, "2 more pizzas"))
    assert fake_repository.get_cart_item_quantity("test-session", "pizza") == 3


def test_quantity_exceeding_stock_is_skipped(fake_llm, fake_repository, base_state):
    fake_llm.extraction_result = RequestedItems(items=[RequestedItem(item="pizza", quantity=30)])
    agent.add_item(_state(base_state, "30 pizzas"))
    assert fake_repository.get_cart_items("test-session") == []
