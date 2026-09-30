from langchain_core.messages import HumanMessage

import agent


def _state(base_state, unclear_count):
    return {
        **base_state,
        "messages": [HumanMessage(content="tell me a joke")],
        "unclear_count": unclear_count,
    }


def test_first_unclear_message_asks_and_does_not_close(fake_llm, fake_repository, base_state):
    result = agent.clarify(_state(base_state, unclear_count=0))
    assert result["unclear_count"] == 1
    assert result.get("status") != "regretted"
    assert result["messages"]


def test_second_unclear_message_asks_and_does_not_close(fake_llm, fake_repository, base_state):
    result = agent.clarify(_state(base_state, unclear_count=1))
    assert result["unclear_count"] == 2
    assert result.get("status") != "regretted"


def test_third_consecutive_unclear_message_closes_session(fake_llm, fake_repository, base_state):
    result = agent.clarify(_state(base_state, unclear_count=2))
    assert result["unclear_count"] == agent.MAX_UNCLEAR_ATTEMPTS
    assert result["status"] == "regretted"
    assert result["messages"]
