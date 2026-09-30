"""
SQLAlchemy models - the 6 tables that replace Redis as the source of
truth for menu, sessions, carts, and orders:

    menu_items, sessions, carts, cart_items, orders, order_items

(restaurants/users/payments/agent_runs were considered and deliberately
left out for now - see prompt_implementation.md's dated instructions for
why. They're cheap to add later if multi-restaurant support, real auth,
payments, or SQL-queryable run history become actual requirements.)
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MenuItem(Base):
    __tablename__ = "menu_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    price: Mapped[float] = mapped_column(Numeric(10, 2))
    stock: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(20), default="browsing", index=True)
    query_count: Mapped[int] = mapped_column(Integer, default=0)
    cook_retry_count: Mapped[int] = mapped_column(Integer, default=0)
    unclear_count: Mapped[int] = mapped_column(Integer, default=0)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    cart: Mapped["Cart | None"] = relationship(back_populates="session", uselist=False)
    orders: Mapped[list["Order"]] = relationship(back_populates="session")


class Cart(Base):
    __tablename__ = "carts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    session: Mapped[Session] = relationship(back_populates="cart")
    items: Mapped[list["CartItem"]] = relationship(back_populates="cart", cascade="all, delete-orphan")


class CartItem(Base):
    __tablename__ = "cart_items"
    __table_args__ = (UniqueConstraint("cart_id", "menu_item_id", name="uq_cart_item"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cart_id: Mapped[int] = mapped_column(ForeignKey("carts.id"), index=True)
    menu_item_id: Mapped[int] = mapped_column(ForeignKey("menu_items.id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer)

    cart: Mapped[Cart] = relationship(back_populates="items")
    menu_item: Mapped[MenuItem] = relationship()


class Order(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="ordered", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    session: Mapped[Session] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    __tablename__ = "order_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    order_id: Mapped[int] = mapped_column(ForeignKey("orders.id"), index=True)
    menu_item_id: Mapped[int] = mapped_column(ForeignKey("menu_items.id"), index=True)
    quantity: Mapped[int] = mapped_column(Integer)
    price_at_order: Mapped[float] = mapped_column(Numeric(10, 2))

    order: Mapped[Order] = relationship(back_populates="items")
    menu_item: Mapped[MenuItem] = relationship()
