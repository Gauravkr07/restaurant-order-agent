"""
Post-checkout fan-out: after an order is placed, four independent
follow-up actions run as Celery tasks instead of blocking the checkout
HTTP response. None of these have a real integration yet (no email/SMS
provider, no restaurant-facing dashboard, no invoicing system) - each is
a stub that logs what it would do and simulates a short delay, so the
Celery plumbing itself (broker, worker, fan-out via group(), retries) is
real and exercised end-to-end even though the business action is a
placeholder. Swap the body of each task for a real integration later
without changing how they're invoked.
"""
import time

from celery import group

from helpers.logging_config import get_logger
from tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(name="tasks.send_confirmation", bind=True, max_retries=3, default_retry_delay=5)
def send_confirmation(self, order_id: int) -> dict:
    logger.info("send_confirmation_stub", extra={"order_id": order_id})
    time.sleep(0.5)  # simulated I/O (e.g. calling an email/SMS provider)
    return {"order_id": order_id, "task": "send_confirmation", "status": "sent"}


@celery_app.task(name="tasks.notify_restaurant", bind=True, max_retries=3, default_retry_delay=5)
def notify_restaurant(self, order_id: int) -> dict:
    logger.info("notify_restaurant_stub", extra={"order_id": order_id})
    time.sleep(0.5)  # simulated I/O (e.g. pushing to a kitchen display system)
    return {"order_id": order_id, "task": "notify_restaurant", "status": "notified"}


@celery_app.task(name="tasks.generate_invoice", bind=True, max_retries=3, default_retry_delay=5)
def generate_invoice(self, order_id: int) -> dict:
    logger.info("generate_invoice_stub", extra={"order_id": order_id})
    time.sleep(0.5)  # simulated I/O (e.g. calling a billing/invoicing service)
    return {"order_id": order_id, "task": "generate_invoice", "status": "generated"}


@celery_app.task(name="tasks.update_delivery_status", bind=True, max_retries=3, default_retry_delay=5)
def update_delivery_status(self, order_id: int) -> dict:
    logger.info("update_delivery_status_stub", extra={"order_id": order_id})
    time.sleep(0.5)  # simulated I/O (e.g. calling a delivery/logistics provider)
    return {"order_id": order_id, "task": "update_delivery_status", "status": "updated"}


@celery_app.task(name="tasks.process_completed_order")
def process_completed_order(order_id: int) -> str:
    """Fans the four follow-up tasks out in parallel (Celery group, not a
    sequential chain) - they're independent of each other, so there's no
    reason to make one wait on another. Returns the group result's id;
    callers that want to know when all four finish can poll it via
    celery_app.AsyncResult(group_id) or celery.result.GroupResult."""
    job = group(
        send_confirmation.s(order_id),
        notify_restaurant.s(order_id),
        generate_invoice.s(order_id),
        update_delivery_status.s(order_id),
    )
    result = job.apply_async()
    result.save()  # so the group can be looked up later by id
    logger.info("process_completed_order_dispatched", extra={"order_id": order_id, "group_id": result.id})
    return result.id
