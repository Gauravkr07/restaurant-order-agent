from langchain_core.messages import HumanMessage

import agent
from helpers.models import RequestedItem, RequestedItems


def _state(base_state, message):
    return {**base_state, "messages": [HumanMessage(content=message)]}


def test_removing_partial_quantity_updates_line(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 3)
    fake_llm.extraction_result = RequestedItems(items=[RequestedItem(item="pizza", quantity=1)])
    agent.remove_item(_state(base_state, "remove 1 pizza"))
    assert fake_repository.get_cart_item_quantity("test-session", "pizza") == 2


def test_removing_full_quantity_deletes_line(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 3)
    fake_llm.extraction_result = RequestedItems(items=[RequestedItem(item="pizza", quantity=3)])
    agent.remove_item(_state(base_state, "remove 3 pizza"))
    assert fake_repository.get_cart_items("test-session") == []


def test_removing_more_than_available_leaves_cart_unchanged_and_reports(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 3)
    fake_llm.extraction_result = RequestedItems(items=[RequestedItem(item="pizza", quantity=10)])
    result = agent.remove_item(_state(base_state, "remove 10 pizza"))
    assert any("Only 3" in m for m in result["messages"])
    assert fake_repository.get_cart_item_quantity("test-session", "pizza") == 3


def test_removing_item_not_in_cart_reports_without_crashing(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 3)
    fake_llm.extraction_result = RequestedItems(items=[RequestedItem(item="burger", quantity=1)])
    result = agent.remove_item(_state(base_state, "remove 1 burger"))
    assert any("not in cart" in m for m in result["messages"])
    assert fake_repository.get_cart_item_quantity("test-session", "pizza") == 3


def test_removing_nothing_extracted_still_returns_a_reply(fake_llm, fake_repository, base_state):
    fake_repository.add_cart_item("test-session", "pizza", 3)
    fake_llm.extraction_result = RequestedItems(items=[])
    result = agent.remove_item(_state(base_state, "remove some stuff"))
    assert result["messages"]
    assert fake_repository.get_cart_item_quantity("test-session", "pizza") == 3
