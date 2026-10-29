"""
Order lifecycle tools - checkout, tracking, cancellation, refunds. Each
docstring is what the LLM reads to decide when to call the tool, so keep
them precise about what it does and does not do.
"""
from langchain_core.tools import tool

from app.domain import repository


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


@tool
def request_refund_for_order(order_id: int) -> dict | None:
    """Request a refund for an order that has already been delivered
    (or one that failed during cooking). Returns None if the order id
    doesn't exist or isn't in a refundable state (e.g. it's still
    cooking - use cancel_order for an order that hasn't started cooking
    yet - or a refund was already requested/resolved for it).

    Refunds up to ₹100 are approved immediately (returned status:
    "refunded"). Larger refunds need a human to review them first
    (returned status: "pending_refund_approval") - tell the customer
    it's being reviewed and will be handled shortly, don't imply it's
    instant."""
    return repository.request_refund(order_id)


ORDER_TOOLS = [checkout, cancel_order, get_order_status, request_refund_for_order]
