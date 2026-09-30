"""
LangChain tools wrapping db/repository.py, so an LLM can call these
directly (via bind_tools/tool-calling) instead of the app hardcoding
which repository function runs for a given intent. Phase 1 of moving
toward a tool-based agent - see prompt_implementation.md's "Implementing
Agentic AI" section.

Each tool's docstring is what the LLM reads to decide when to call it,
so keep them precise about what the tool does and does not do.
"""
from langchain_core.tools import tool

from db import repository


@tool
def search_menu(query: str = "") -> list[dict]:
    """List active menu items (name, price, stock). If `query` is given,
    only items whose name contains it are returned (case-insensitive).
    Use this to see what's available, or to check spelling/existence of
    an item the customer mentioned."""
    items = repository.get_active_menu_items()
    if not query:
        return items
    query_lower = query.lower()
    return [item for item in items if query_lower in item["name"].lower()]


@tool
def get_menu_item(name: str) -> dict | None:
    """Get one menu item's price and current stock by exact name.
    Returns None if the item doesn't exist or isn't active."""
    return repository.get_menu_item(name)


@tool
def add_to_cart(session_id: str, item_name: str, quantity: int) -> dict:
    """Add `quantity` of `item_name` to the customer's cart for this
    session. If the item is already in the cart, the quantity is summed,
    not replaced. Does NOT check stock - call get_menu_item first to
    verify the item exists and has enough stock before calling this."""
    repository.add_cart_item(session_id, item_name, quantity)
    return {"cart": repository.get_cart_items(session_id)}


@tool
def remove_from_cart(session_id: str, item_name: str, quantity: int) -> dict:
    """Remove `quantity` of `item_name` from the customer's cart.
    Returns {"outcome": ...} where outcome is one of: "not_in_cart" (item
    isn't in the cart), "insufficient" (asked for more than is in the
    cart - nothing was changed), "removed" (line deleted), "updated"
    (quantity reduced)."""
    outcome = repository.remove_cart_item(session_id, item_name, quantity)
    return {"outcome": outcome, "cart": repository.get_cart_items(session_id)}


@tool
def get_cart(session_id: str) -> list[dict]:
    """Get the customer's current cart contents (item, quantity, price
    per line). Returns an empty list if the cart is empty."""
    return repository.get_cart_items(session_id)


@tool
def checkout(session_id: str) -> dict | None:
    """Place an order from everything currently in the customer's cart.
    Snapshots each line's price and empties the cart. Returns None if
    the cart was empty (nothing to order)."""
    return repository.create_order_from_cart(session_id)


@tool
def cancel_order(order_id: int) -> dict | None:
    """Cancel an order, but only if it hasn't started cooking yet.
    Returns None if the order doesn't exist or is already past the point
    where it can be cancelled (use the refund flow for orders that failed
    during cooking instead)."""
    return repository.cancel_order(order_id)


@tool
def get_order_status(order_id: int) -> dict | None:
    """Get an order's current status and items. Returns None if the
    order id doesn't exist."""
    return repository.get_order(order_id)


ALL_TOOLS = [
    search_menu,
    get_menu_item,
    add_to_cart,
    remove_from_cart,
    get_cart,
    checkout,
    cancel_order,
    get_order_status,
]
