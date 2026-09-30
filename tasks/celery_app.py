"""
Celery app instance, using Redis as both broker and result backend (the
same Redis already used for session counters and the checkout lock - see
service.py - so no new infrastructure beyond what's already running).
"""
import os

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "restaurant_order_agent",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.order_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
