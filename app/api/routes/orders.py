"""
Order tracking, idempotent checkout, and human-in-the-loop refund
approval.
"""
from fastapi import APIRouter, Header, HTTPException
from redis.exceptions import LockError

from app.api.schemas import (
    CheckoutRequest,
    CheckoutResponse,
    OrderStatusResponse,
    RefundDecisionResponse,
)
from app.core.config import settings
from app.core.logging import current_session_id, get_logger
from app.domain import repository
from app.domain.repository import OutOfStockError
from app.domain.session_store import redis_client
from app.tasks.order_tasks import process_completed_order

logger = get_logger(__name__)
router = APIRouter(tags=["orders"])


@router.get("/orders/{order_id}")
def get_order_endpoint(order_id: int) -> OrderStatusResponse:
    """Public order tracking - no auth/session check, same as the rest
    of this app (no real accounts exist yet). Anyone with the order
    number can look it up, matching how most small delivery apps let
    you "track" an order by number alone."""
    order = repository.get_order(order_id)
    if order is None:
        raise HTTPException(status_code=404, detail="No order with that id.")
    return OrderStatusResponse(
        order_id=order["order_id"], status=order["status"], items=order["items"]
    )


@router.post("/checkout")
def checkout_endpoint(
    req: CheckoutRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
) -> CheckoutResponse:
    """The literal "POST /checkout with an Idempotency-Key header"
    contract, separate from the chat-based checkout node - for a client
    (e.g. a "Pay" button) that can generate and resend the same key on
    retry/double-click."""
    current_session_id.set(req.session_id)
    if repository.get_session(req.session_id) is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")

    # Redis distributed lock: the DB-level protections (idempotency_key
    # UNIQUE constraint + SELECT ... FOR UPDATE on the cart row) are what
    # actually guarantee correctness even without this lock - it exists
    # to fail fast and avoid making two concurrent requests both sit
    # blocked on the same Postgres row lock for no reason, and to give a
    # clear timeout/error if something is stuck.
    lock = redis_client.lock(
        f"checkout-lock:{idempotency_key}",
        timeout=settings.checkout_lock_timeout_seconds,
        blocking_timeout=settings.checkout_lock_wait_seconds,
    )
    if not lock.acquire(blocking=True):
        logger.warning("checkout_lock_timeout", extra={"idempotency_key": idempotency_key})
        raise HTTPException(
            status_code=409,
            detail="Another checkout with this Idempotency-Key is already in "
            "progress. Please retry shortly.",
        )
    try:
        order, created = repository.create_order_idempotent(req.session_id, idempotency_key)
    except OutOfStockError as exc:
        # Nothing was ordered and the cart is untouched - the customer
        # can adjust it and retry, so this is a 409, not a 500.
        logger.warning("checkout_out_of_stock", extra={"shortfalls": exc.shortfalls})
        detail = "; ".join(
            f"{s['item']}: only {s['available']} left (you wanted {s['wanted']})"
            for s in exc.shortfalls
        )
        raise HTTPException(
            status_code=409, detail=f"Some items just went out of stock - {detail}."
        ) from exc
    finally:
        try:
            lock.release()
        except LockError:
            pass  # lock already expired (timeout) - fine, nothing to release

    if order is None:
        raise HTTPException(status_code=400, detail="Cart is empty - nothing to check out.")

    logger.info(
        "checkout_endpoint_result",
        extra={
            "order_id": order["order_id"],
            "was_created": created,
            "idempotency_key": idempotency_key,
        },
    )
    if created:
        # Only fan out on a genuinely new order - an idempotent replay
        # (created=False) must NOT re-send confirmations/invoices, or
        # the customer's three "PAY" clicks would produce three emails.
        process_completed_order.delay(order["order_id"])
    return CheckoutResponse(
        order_id=order["order_id"],
        status=order["status"],
        items=order["items"],
        created=created,
    )


# --- human-in-the-loop refund approval ---
# No admin UI yet - these are plain endpoints a human calls directly,
# e.g. via curl/Postman, after reviewing a large refund. See
# repository.request_refund for the >100 threshold that routes an order
# here in the first place.


@router.post("/orders/{order_id}/approve-refund")
def approve_refund_endpoint(order_id: int) -> RefundDecisionResponse:
    order = repository.approve_refund(order_id)
    if order is None:
        raise HTTPException(
            status_code=404, detail="No order with that id is awaiting refund approval."
        )
    logger.info("refund_approved", extra={"order_id": order_id})
    return RefundDecisionResponse(order_id=order["order_id"], status=order["status"])


@router.post("/orders/{order_id}/reject-refund")
def reject_refund_endpoint(order_id: int) -> RefundDecisionResponse:
    order = repository.reject_refund(order_id)
    if order is None:
        raise HTTPException(
            status_code=404, detail="No order with that id is awaiting refund approval."
        )
    logger.info("refund_rejected", extra={"order_id": order_id})
    return RefundDecisionResponse(order_id=order["order_id"], status=order["status"])
