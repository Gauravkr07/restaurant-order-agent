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

# A managed Redis reached over TLS uses the rediss:// scheme (Render,
# Heroku, Upstash). Celery/kombu will not negotiate TLS for such a URL
# unless it is given an ssl config, and fails to connect without one.
# Certificates are verified by default. If a provider terminates TLS
# with an internal CA that isn't in the container's trust store, supply
# that CA via REDIS_SSL_CA_CERTS rather than turning verification off.
if settings.redis_url.startswith("rediss://"):
    import ssl

    ssl_options = {"ssl_cert_reqs": ssl.CERT_REQUIRED}
    if settings.redis_ssl_ca_certs:
        ssl_options["ssl_ca_certs"] = settings.redis_ssl_ca_certs
    celery_app.conf.broker_use_ssl = ssl_options
    celery_app.conf.redis_backend_use_ssl = ssl_options
