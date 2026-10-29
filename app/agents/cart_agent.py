"""
Cart agents: adding items to the cart and removing them.

Both are tool-calling agents - the LLM is given the menu/cart tools and
decides itself which to call and when, instead of a hardcoded
extract-then-check-then-mutate pipeline.
"""
from app.agents import prompts
from app.agents.runtime import invoke_structured, run_tool_loop
from app.agents.schemas import AddItemVerdict, RequestedItem, UnavailableItem, UnavailableItems
from app.tools import add_to_cart, get_cart, get_menu_item, remove_from_cart, search_menu

ADD_ITEM_TOOLS = [search_menu, get_menu_item, add_to_cart]
REMOVE_ITEM_TOOLS = [remove_from_cart, get_cart]


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
    return AddItemVerdict(added=added, unavailable=extract_unavailable_items(final_summary))


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
