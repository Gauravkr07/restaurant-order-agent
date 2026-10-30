"""
run_tool_loop's failure handling (G1).

A single bad tool call used to end the whole conversation turn. Each of
these cases must instead be handed back to the model as a tool result,
so it can correct itself within the remaining rounds.
"""
import groq
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from app.agents import runtime


@tool
def working_tool(value: str) -> dict:
    """Returns its input."""
    return {"echo": value}


@tool
def exploding_tool(value: str) -> dict:
    """Always raises."""
    raise RuntimeError("tool blew up")


class ScriptedToolLLM:
    """Returns each queued response in turn. A response is either an
    AIMessage or an exception to raise."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        response = self.responses.pop(0) if self.responses else AIMessage(content="done")
        if isinstance(response, BaseException):
            raise response
        return response


class StubLLM:
    """Stands in for the module-level llm; ChatGroq is a pydantic model,
    so a method can't be patched onto the real instance."""

    def __init__(self, scripted):
        self._scripted = scripted

    def bind_tools(self, tools):
        return self._scripted


def _bind(monkeypatch, scripted):
    monkeypatch.setattr(runtime, "llm", StubLLM(scripted))


def _call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_unknown_tool_name_does_not_raise(monkeypatch):
    """A hallucinated tool name used to raise StopIteration."""
    scripted = ScriptedToolLLM([
        _call("no_such_tool", {"value": "x"}),
        AIMessage(content="recovered"),
    ])
    _bind(monkeypatch, scripted)

    result = runtime.run_tool_loop([working_tool], "sys", "msg")

    assert result == "recovered"
    assert scripted.calls == 2


def test_failing_tool_is_reported_back_not_raised(monkeypatch):
    scripted = ScriptedToolLLM([
        _call("exploding_tool", {"value": "x"}),
        AIMessage(content="handled the failure"),
    ])
    _bind(monkeypatch, scripted)

    result = runtime.run_tool_loop([exploding_tool], "sys", "msg")

    assert result == "handled the failure"


def test_failing_tool_does_not_count_as_a_successful_call(monkeypatch):
    """on_tool_call feeds the cart verdict, so it must only fire when the
    tool actually succeeded."""
    scripted = ScriptedToolLLM([
        _call("exploding_tool", {"value": "x"}),
        AIMessage(content="done"),
    ])
    _bind(monkeypatch, scripted)
    seen = []

    runtime.run_tool_loop(
        [exploding_tool], "sys", "msg", on_tool_call=lambda n, a: seen.append(n)
    )

    assert seen == []


def test_successful_tool_is_recorded(monkeypatch):
    scripted = ScriptedToolLLM([
        _call("working_tool", {"value": "x"}),
        AIMessage(content="done"),
    ])
    _bind(monkeypatch, scripted)
    seen = []

    runtime.run_tool_loop(
        [working_tool], "sys", "msg", on_tool_call=lambda n, a: seen.append(n)
    )

    assert seen == ["working_tool"]


def test_transient_model_error_is_retried(monkeypatch):
    """The two Groq flakes seen in production: malformed tool-call JSON
    and "Tool choice is required, but model did not call a tool"."""
    scripted = ScriptedToolLLM([
        groq.APIError("Failed to parse tool call arguments as JSON", request=None, body=None),
        AIMessage(content="second attempt worked"),
    ])
    _bind(monkeypatch, scripted)

    result = runtime.run_tool_loop([working_tool], "sys", "msg")

    assert result == "second attempt worked"
    assert scripted.calls == 2


def test_persistent_model_error_returns_empty_rather_than_raising(monkeypatch):
    err = groq.APIError("Tool choice is required", request=None, body=None)
    scripted = ScriptedToolLLM([err, err])
    _bind(monkeypatch, scripted)

    result = runtime.run_tool_loop([working_tool], "sys", "msg")

    assert result == ""


def test_loop_stops_at_max_rounds(monkeypatch):
    """A model that calls a tool forever must be bounded."""
    scripted = ScriptedToolLLM(
        [_call("working_tool", {"value": "x"}, f"c{i}") for i in range(20)]
    )
    _bind(monkeypatch, scripted)

    runtime.run_tool_loop([working_tool], "sys", "msg")

    assert scripted.calls == runtime.MAX_TOOL_CALL_ROUNDS


# --- the extraction call the add flow can now skip or survive ---

def test_add_item_skips_extraction_when_everything_was_added(monkeypatch):
    """The expensive, flake-prone structured-output call must not run at
    all when the tool record already accounts for every requested item."""
    from app.agents import cart_agent

    monkeypatch.setattr(cart_agent, "run_tool_loop", lambda *a, **k: _added_both(k))
    called = []
    monkeypatch.setattr(
        cart_agent, "extract_unavailable_items", lambda s: called.append(s) or []
    )

    verdict = cart_agent.run_add_item_agent("s1", "2 pizzas and a coke")

    assert called == []
    assert len(verdict.added) == 2
    assert verdict.unavailable == []


def _added_both(kwargs):
    """Simulate the loop adding both requested items via on_tool_call."""
    hook = kwargs.get("on_tool_call")
    if hook:
        hook("add_to_cart", {"item_name": "pizza", "quantity": 2})
        hook("add_to_cart", {"item_name": "coke", "quantity": 1})
    return "Added 2 x pizza and 1 x coke."


def test_add_item_survives_extraction_failure(monkeypatch):
    """Extraction flaking must not lose the turn - the cart is already
    correct, so the verdict is built from the tool record instead."""
    from app.agents import cart_agent

    # One item added but two asked for, so the skip does not apply and
    # extraction is attempted; it then flakes.
    def added_one(*a, **k):
        hook = k.get("on_tool_call")
        if hook:
            hook("add_to_cart", {"item_name": "coke", "quantity": 1})
        return "Added 1 x coke."

    monkeypatch.setattr(cart_agent, "run_tool_loop", added_one)

    def boom(_summary):
        raise groq.APIError("Tool choice is required", request=None, body=None)

    monkeypatch.setattr(cart_agent, "extract_unavailable_items", boom)

    verdict = cart_agent.run_add_item_agent("s1", "2 pizzas and a coke")

    # The turn survives: the cart content is reported, and the shortfall
    # is described from the model's own summary rather than raising.
    assert len(verdict.added) == 1
    assert len(verdict.unavailable) == 1


def test_add_item_reports_shortfall_when_fewer_items_were_added(monkeypatch):
    """Extraction flaked AND the tool record shows a gap - the customer
    must still be told something could not be added."""
    from app.agents import cart_agent

    def added_one(*a, **k):
        hook = k.get("on_tool_call")
        if hook:
            hook("add_to_cart", {"item_name": "coke", "quantity": 1})
        return "Added 1 x coke. Pizza is out of stock."

    monkeypatch.setattr(cart_agent, "run_tool_loop", added_one)

    def boom(_summary):
        raise groq.APIError("Tool choice is required", request=None, body=None)

    monkeypatch.setattr(cart_agent, "extract_unavailable_items", boom)

    verdict = cart_agent.run_add_item_agent("s1", "2 pizzas and a coke")

    assert len(verdict.added) == 1
    assert len(verdict.unavailable) == 1
    assert "out of stock" in verdict.unavailable[0].reason


def test_extraction_failure_does_not_invent_a_shortfall(monkeypatch):
    """Both items added but extraction still attempted and flaking (a
    single-item phrasing the heuristic counts as one): the customer must
    not be told something failed when nothing did."""
    from app.agents import cart_agent

    def added_two(*a, **k):
        hook = k.get("on_tool_call")
        if hook:
            hook("add_to_cart", {"item_name": "pizza", "quantity": 2})
        return "Added 2 x pizza."

    monkeypatch.setattr(cart_agent, "run_tool_loop", added_two)
    monkeypatch.setattr(
        cart_agent, "extract_unavailable_items",
        lambda s: (_ for _ in ()).throw(
            groq.APIError("Tool choice is required", request=None, body=None)
        ),
    )

    verdict = cart_agent.run_add_item_agent("s1", "2 pizzas")

    assert len(verdict.added) == 1
    assert verdict.unavailable == []
