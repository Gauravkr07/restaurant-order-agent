"""LangChain tools wrapping app/domain/repository.py, grouped by aggregate."""
from app.tools.cart import CART_TOOLS, add_to_cart, get_cart, remove_from_cart
from app.tools.menu import MENU_TOOLS, get_menu_item, search_menu
from app.tools.order import (
    ORDER_TOOLS,
    cancel_order,
    checkout,
    get_order_status,
    request_refund_for_order,
)

ALL_TOOLS = [*MENU_TOOLS, *CART_TOOLS, *ORDER_TOOLS]

__all__ = [
    "ALL_TOOLS", "MENU_TOOLS", "CART_TOOLS", "ORDER_TOOLS",
    "search_menu", "get_menu_item",
    "add_to_cart", "remove_from_cart", "get_cart",
    "checkout", "cancel_order", "get_order_status", "request_refund_for_order",
]
