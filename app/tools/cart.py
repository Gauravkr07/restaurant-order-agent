"""
Cart mutation and inspection tools. Each docstring is what the LLM reads
to decide when to call the tool, so keep them precise about what it does
and does not do.
"""
from langchain_core.tools import tool

from app.domain import repository


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


CART_TOOLS = [add_to_cart, remove_from_cart, get_cart]
