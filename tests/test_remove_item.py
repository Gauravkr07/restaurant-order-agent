from langchain_core.messages import HumanMessage

import agent
from helpers.models import Cart, RequestedItem, RequestedItems


def _state(base_state, message, cart):
    return {**base_state, "messages": [HumanMessage(content=message)], "cart": cart}


def test_removing_partial_quantity_updates_line(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(
        items=[RequestedItem(item="pizza", quantity=1)]
    )
    cart = [Cart(item="pizza", quantity=3, price=200.0)]
    result = agent.remove_item(_state(base_state, "remove 1 pizza", cart=cart))
    assert len(result["cart"]) == 1
    assert result["cart"][0].quantity == 2


def test_removing_full_quantity_deletes_line(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(
        items=[RequestedItem(item="pizza", quantity=3)]
    )
    cart = [Cart(item="pizza", quantity=3, price=200.0)]
    result = agent.remove_item(_state(base_state, "remove 3 pizza", cart=cart))
    assert result["cart"] == []


def test_removing_more_than_available_leaves_cart_unchanged_and_reports(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(
        items=[RequestedItem(item="pizza", quantity=10)]
    )
    cart = [Cart(item="pizza", quantity=3, price=200.0)]
    result = agent.remove_item(_state(base_state, "remove 10 pizza", cart=cart))
    assert any("Only 3" in m for m in result["messages"])
    assert result["cart"] == [Cart(item="pizza", quantity=3, price=200.0)]


def test_removing_item_not_in_cart_reports_without_crashing(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(
        items=[RequestedItem(item="burger", quantity=1)]
    )
    cart = [Cart(item="pizza", quantity=3, price=200.0)]
    result = agent.remove_item(_state(base_state, "remove 1 burger", cart=cart))
    assert any("not in cart" in m for m in result["messages"])
    assert result["cart"] == cart


def test_removing_nothing_extracted_still_returns_a_reply(fake_llm, base_state):
    fake_llm.extraction_result = RequestedItems(items=[])
    cart = [Cart(item="pizza", quantity=3, price=200.0)]
    result = agent.remove_item(_state(base_state, "remove some stuff", cart=cart))
    assert result["messages"]
    assert result["cart"] == cart
