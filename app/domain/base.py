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

engine = create_engine(settings.database_url, pool_pre_ping=True)
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
