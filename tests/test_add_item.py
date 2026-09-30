from langchain_core.messages import HumanMessage

import agent
from helpers.models import Cart, RequestedItem, RequestedItems


def _state(base_state, message, cart):
    return {**base_state, "messages": [HumanMessage(content=message)], "cart": cart}


def test_adding_new_item_creates_cart_line(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(
        items=[RequestedItem(item="burger", quantity=1)]
    )
    result = agent.add_item(_state(base_state, "1 burger", cart=[]))
    assert len(result["cart"]) == 1
    line = result["cart"][0]
    assert line.item == "burger"
    assert line.quantity == 1
    assert line.price == 150.0
    assert result["status"] == "cart"


def test_adding_more_of_existing_item_sums_quantity_not_replaces(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(
        items=[RequestedItem(item="pizza", quantity=2)]
    )
    existing_cart = [Cart(item="pizza", quantity=1, price=200.0)]
    result = agent.add_item(_state(base_state, "2 more pizzas", cart=existing_cart))
    assert len(result["cart"]) == 1
    assert result["cart"][0].quantity == 3


def test_quantity_exceeding_stock_is_skipped(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(
        items=[RequestedItem(item="pizza", quantity=30)]  # pizza stock is 20
    )
    result = agent.add_item(_state(base_state, "30 pizzas", cart=[]))
    assert result["cart"] == []
