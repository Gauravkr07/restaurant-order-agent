"""
Menu lookup tools. Each docstring is what the LLM reads to decide when
to call the tool, so keep them precise about what it does and does not do.
"""
from langchain_core.tools import tool

from app.domain import repository


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


MENU_TOOLS = [search_menu, get_menu_item]
