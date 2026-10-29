"""
Which order statuses a refund may be requested against (G7).

A customer can only be refunded for food they actually received. The
kitchen's own failure path is the one exception, and it has to ask for
it explicitly.
"""
import pytest

from app.domain.repository import (
    KITCHEN_FAILURE_REFUNDABLE_STATUSES,
    REFUNDABLE_STATUSES,
)


def test_customer_refund_allows_only_delivered_orders():
    assert REFUNDABLE_STATUSES == ("delivered",)


def test_ordered_is_not_customer_refundable():
    """The G7 regression: an "ordered" order has been paid for but not
    delivered, so refunding it pays out for food that may still arrive."""
    assert "ordered" not in REFUNDABLE_STATUSES


def test_kitchen_failure_may_refund_undelivered_orders():
    """cooking_stage refunds an order whose DB status is still "ordered",
    because the customer paid and the kitchen produced nothing."""
    assert "ordered" in KITCHEN_FAILURE_REFUNDABLE_STATUSES
    assert "delivered" in KITCHEN_FAILURE_REFUNDABLE_STATUSES


@pytest.mark.parametrize(
    "status", ["cancelled", "refunded", "pending_refund_approval", "refund_rejected"]
)
def test_resolved_orders_are_never_refundable(status):
    assert status not in REFUNDABLE_STATUSES
    assert status not in KITCHEN_FAILURE_REFUNDABLE_STATUSES


def test_fake_repository_refuses_customer_refund_on_ordered(fake_repository):
    fake_repository.add_cart_item("s1", "coke", 1)
    order = fake_repository.create_order_from_cart("s1")
    assert order["status"] == "ordered"

    assert fake_repository.request_refund(order["order_id"]) is None


def test_fake_repository_allows_kitchen_refund_on_ordered(fake_repository):
    fake_repository.add_cart_item("s1", "coke", 1)
    order = fake_repository.create_order_from_cart("s1")

    refunded = fake_repository.request_refund(order["order_id"], kitchen_failure=True)

    assert refunded is not None
    assert refunded["status"] == "refunded"


def test_customer_refund_allowed_once_delivered(fake_repository):
    fake_repository.add_cart_item("s1", "coke", 1)
    order = fake_repository.create_order_from_cart("s1")
    fake_repository.update_order_status(order["order_id"], "delivered")

    refunded = fake_repository.request_refund(order["order_id"])

    assert refunded is not None
    assert refunded["status"] == "refunded"
