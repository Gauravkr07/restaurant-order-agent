"""
Celery app instance, using Redis as both broker and result backend (the
same Redis already used for session counters and the checkout lock - see
app/domain/session_store.py - so no new infrastructure beyond what's already running).
"""
from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "restaurant_order_agent",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.tasks.order_tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
)
