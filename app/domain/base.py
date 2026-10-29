"""
SQLAlchemy engine/session setup. PostgreSQL is now the source of truth
for menu, sessions, carts, and orders - Redis is no longer used for any
of that (see README "Data layer" section for the reasoning). Redis is
used for session counters (app/domain/session_store.py), the checkout
lock, and as the Celery broker.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.database_url,
    # Verify a pooled connection is still alive before handing it out.
    # Essential against a managed/serverless Postgres (Neon and friends),
    # which closes idle connections and can suspend a branch entirely -
    # without this, the first request after a quiet spell fails.
    pool_pre_ping=True,
    # Drop and rebuild connections older than this, so the pool can't
    # fill with handles the server has already discarded.
    pool_recycle=settings.db_pool_recycle_seconds,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db_session():
    """FastAPI-style dependency: yields a session, always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
