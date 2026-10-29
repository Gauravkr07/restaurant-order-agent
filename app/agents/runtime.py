"""
Shared machinery every agent uses: the tool-calling loop, the
structured-output retry wrapper, and reply composition.

This is the layer that was previously inlined and duplicated throughout
agent.py - in particular fourteen near-identical llm.invoke([...]) blocks
that each hand-rolled the same ("system", persona) / ("human", facts) pair.
"""
import json

import groq
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.core.llm import llm
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_TOOL_CALL_ROUNDS = 6
STRUCTURED_OUTPUT_RETRY_ATTEMPTS = 2


def compose_reply(instruction: str, facts: str) -> str:
    """Turn already-validated facts into a customer-facing sentence.

    The LLM is only ever asked to WORD the reply here - never to decide
    it. Everything in `facts` has already been established by Python and
    the repository layer, which is why these prompts carry no untrusted
    input notice.
    """
    return llm.invoke([("system", instruction), ("human", facts)])


def invoke_structured(schema, instruction: str, content: str):
    """Run a structured-output call, retrying if the model flakes.

    Retries on:
    - groq.GroqError: Groq occasionally errors with "Tool choice is
      required, but model did not call a tool" instead of returning
      structured output.
    - pydantic.ValidationError: the model can return a value that fails
      the schema's validation (e.g. an out-of-range quantity) - worth one
      retry before giving up, since it's usually a one-off glitch.
    """
    extractor = llm.with_structured_output(schema)
    messages = [("system", instruction), ("human", content)]
    last_error: Exception | None = None
    for attempt in range(STRUCTURED_OUTPUT_RETRY_ATTEMPTS):
        try:
            return extractor.invoke(messages)
        except (groq.GroqError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "structured_output_retry",
                extra={
                    "schema": schema.__name__,
                    "attempt": attempt + 1,
                    "max_attempts": STRUCTURED_OUTPUT_RETRY_ATTEMPTS,
                    "error_type": type(exc).__name__,
                },
            )
    logger.error(
        "structured_output_failed",
        extra={
            "schema": schema.__name__,
            "attempts": STRUCTURED_OUTPUT_RETRY_ATTEMPTS,
            "error_type": type(last_error).__name__,
        },
    )
    assert last_error is not None
    raise last_error


def run_tool_loop(tools: list, system_prompt: str, user_message: str, on_tool_call=None) -> str:
    """Shared tool-calling loop: the LLM is bound to `tools` and decides
    itself which to call and when (call tool -> see result -> call
    another tool or finish), up to MAX_TOOL_CALL_ROUNDS. `on_tool_call`,
    if given, is called with (tool_name, tool_args) after each successful
    tool call - used to build a structured record of what happened
    without relying on the model to report it accurately in prose.

    Returns the final plain-text message once the model stops calling
    tools. Callers that need structured data out of that summary should
    run a SEPARATE, tool-free structured-output call on it (see
    cart_agent.extract_unavailable_items) rather than asking for
    structured output in this same context - mixing tool-call history
    with a structured-output request is what triggers Groq's "Tool choice
    is required, but model did not call a tool" error.
    """
    tool_llm = llm.bind_tools(tools)
    messages = [("system", system_prompt), ("human", user_message)]
    tools_by_name = {t.name: t for t in tools}

    ai_message = None
    for _ in range(MAX_TOOL_CALL_ROUNDS):
        ai_message = tool_llm.invoke(messages)
        messages.append(ai_message)
        if not ai_message.tool_calls:
            break
        for call in ai_message.tool_calls:
            tool = tools_by_name[call["name"]]
            result = tool.invoke(call["args"])
            if on_tool_call is not None:
                on_tool_call(call["name"], call["args"])
            messages.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
            )
    else:
        logger.warning("tool_loop_max_rounds_reached", extra={"rounds": MAX_TOOL_CALL_ROUNDS})

    return ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
