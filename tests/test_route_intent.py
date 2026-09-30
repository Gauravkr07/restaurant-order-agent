from langchain_core.messages import HumanMessage

import agent


def _state(base_state, message, **overrides):
    return {**base_state, "messages": [HumanMessage(content=message)], **overrides}


def test_add_item_intent_routes_to_verify_items(fake_llm, base_state):
    fake_llm.intent = "ADD_ITEM"
    result = agent.route_intent(_state(base_state, "2 pizzas please"))
    assert result == "VERIFY_ITEMS"


def test_show_menu_intent_routes_to_show_menu(fake_llm, base_state):
    fake_llm.intent = "SHOW_MENU"
    result = agent.route_intent(_state(base_state, "what's on the menu?"))
    assert result == "SHOW_MENU"


def test_unclear_intent_routes_to_clarify(fake_llm, base_state):
    fake_llm.intent = "UNCLEAR"
    result = agent.route_intent(_state(base_state, "what time do you close?"))
    assert result == "CLARIFY"


def test_unparseable_intent_routes_to_clarify_not_show_menu(fake_llm, base_state):
    fake_llm.intent = "SOMETHING_RANDOM_THE_MODEL_MADE_UP"
    result = agent.route_intent(_state(base_state, "asdkjaslkdj"))
    assert result == "CLARIFY"


def test_cooking_retry_status_bypasses_classification(fake_llm, base_state):
    state = _state(base_state, "is it ready?", status="cooking_retry")
    result = agent.route_intent(state)
    assert result == "COOKING_STAGE"
    assert fake_llm.classify_calls == 0
