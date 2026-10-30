"""
Cart agents: adding items to the cart and removing them.

Both are tool-calling agents - the LLM is given the menu/cart tools and
decides itself which to call and when, instead of a hardcoded
extract-then-check-then-mutate pipeline.
"""
import re

import groq

from app.agents import prompts
from app.agents.runtime import invoke_structured, run_tool_loop
from app.agents.schemas import AddItemVerdict, RequestedItem, UnavailableItem, UnavailableItems
from app.core.logging import get_logger
from app.tools import add_to_cart, get_cart, get_menu_item, remove_from_cart, search_menu

logger = get_logger(__name__)

ADD_ITEM_TOOLS = [search_menu, get_menu_item, add_to_cart]
REMOVE_ITEM_TOOLS = [remove_from_cart, get_cart]


def _requested_item_count(user_message: str) -> int:
    """How many distinct things the customer seems to have asked for.
    Only used to decide whether the extraction call below can be skipped,
    so a rough count is fine - it never decides what goes in the cart."""
    return len(re.findall(r"\b(?:and|,|\+|also|plus)\b|,", user_message)) + 1


def extract_unavailable_items(summary_text: str) -> list[UnavailableItem]:
    """A fresh, tool-free structured-output call (no tool-call/tool-result
    history in context) that pulls the unavailable items and reasons out
    of run_add_item_agent's final plain-text summary. Kept separate from
    the tool-calling messages because asking for structured output in the
    same context as tool-call history is what triggers Groq's "Tool
    choice is required, but model did not call a tool" error."""
    result = invoke_structured(
        UnavailableItems,
        prompts.EXTRACT_UNAVAILABLE,
        summary_text or "(no items were unavailable)",
    )
    return result.unavailable


def run_add_item_agent(session_id: str, user_message: str) -> AddItemVerdict:
    """The LLM is given search_menu/get_menu_item/add_to_cart and decides
    itself which to call and when. It's told explicitly to only call
    add_to_cart for items it has confirmed are in stock. A second
    structured-output call turns the loop's final summary into an
    AddItemVerdict."""
    added: list[RequestedItem] = []

    def track_add(tool_name, tool_args):
        if tool_name == "add_to_cart":
            added.append(RequestedItem(item=tool_args["item_name"], quantity=tool_args["quantity"]))

    final_summary = run_tool_loop(
        ADD_ITEM_TOOLS,
        prompts.add_item_agent(session_id),
        user_message,
        on_tool_call=track_add,
    )

    # Does the tool record already account for everything asked for?
    # Computed once, because both the fast path and the failure path
    # below depend on the same answer.
    everything_added = bool(added) and len(added) >= _requested_item_count(user_message)

    # Skip the extraction round-trip when nothing can be missing: there
    # is nothing for it to find. This is both the slowest call in the add
    # flow and the most failure-prone - with_structured_output is a
    # forced tool call, which is exactly what provokes Groq's "Tool
    # choice is required, but model did not call a tool". The tool record
    # is authoritative here (it is what actually hit the cart), so
    # skipping cannot invent or lose an item.
    if everything_added:
        logger.info("skipped_unavailable_extraction", extra={"added": len(added)})
        return AddItemVerdict(added=added, unavailable=[])

    try:
        unavailable = extract_unavailable_items(final_summary)
    except groq.GroqError:
        # Extraction only shapes the wording of the reply - the cart is
        # already correct either way - so losing the whole turn over it
        # is the wrong trade. Report a shortfall only when the tool
        # record shows one (fewer items added than asked for); if
        # everything was added, treat it as a clean add rather than
        # inventing a failure the customer never had.
        logger.warning("unavailable_extraction_failed", exc_info=True)
        if everything_added:
            unavailable = []
        else:
            unavailable = [
                UnavailableItem(
                    item="some items",
                    reason=final_summary.strip() or "could not be added",
                )
            ]
    return AddItemVerdict(added=added, unavailable=unavailable)


def run_remove_item_agent(session_id: str, user_message: str) -> str:
    """The LLM is given remove_from_cart/get_cart and decides itself what
    to remove. Returns the model's final plain-text summary directly as
    the customer reply - no further structured extraction is needed here
    since remove_item has no branching business logic left to drive
    (unlike the add flow's 3-strike regret path)."""
    return run_tool_loop(
        REMOVE_ITEM_TOOLS,
        prompts.remove_item_agent(session_id),
        f"remove item {user_message}",
    )
