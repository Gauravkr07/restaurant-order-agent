"""
SQLAlchemy engine/session setup. PostgreSQL is now the source of truth
for menu, sessions, carts, and orders - Redis is no longer used for any
of that (see README "Data layer" section for the reasoning). Redis is
still available for cache/rate-limiting/locks if those get built later,
but nothing currently in this app reads or writes Redis.
"""
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://restaurant:restaurant@localhost:5433/restaurant_order_agent",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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
