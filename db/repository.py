"""
The "Tools" layer between agent.py and PostgreSQL. Every function here
opens its own short-lived SQLAlchemy session, does one unit of work,
commits, and closes - agent.py's node functions call these as plain
functions and never see a session or raw SQL.

Returns plain dicts/values (not ORM objects) so callers never accidentally
touch a detached ORM instance after its session has closed.
"""
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from db.base import SessionLocal
from db.models import Cart, CartItem, MenuItem, Order, OrderItem, Session


# ---- Menu -------------------------------------------------------------

def get_active_menu_items() -> list[dict]:
    with SessionLocal() as db:
        rows = db.scalars(
            select(MenuItem).where(MenuItem.is_active.is_(True)).order_by(MenuItem.name)
        ).all()
        return [
            {"name": r.name, "price": float(r.price), "stock": r.stock}
            for r in rows
        ]


def get_menu_item(name: str) -> dict | None:
    with SessionLocal() as db:
        row = db.scalar(
            select(MenuItem).where(MenuItem.name == name, MenuItem.is_active.is_(True))
        )
        if row is None:
            return None
        return {"name": row.name, "price": float(row.price), "stock": row.stock}


# ---- Sessions -----------------------------------------------------------

def get_or_create_session(session_id: str) -> dict:
    with SessionLocal() as db:
        row = db.get(Session, session_id)
        if row is None:
            row = Session(id=session_id)
            db.add(row)
            db.commit()
            db.refresh(row)
        return _session_to_dict(row)


def get_session(session_id: str) -> dict | None:
    with SessionLocal() as db:
        row = db.get(Session, session_id)
        return _session_to_dict(row) if row else None


def update_session(session_id: str, **fields) -> dict:
    with SessionLocal() as db:
        row = db.get(Session, session_id)
        if row is None:
            raise ValueError(f"Unknown session_id: {session_id}")
        for key, value in fields.items():
            setattr(row, key, value)
        row.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(row)
        return _session_to_dict(row)


def _session_to_dict(row: Session) -> dict:
    return {
        "id": row.id,
        "status": row.status,
        "query_count": row.query_count,
        "cook_retry_count": row.cook_retry_count,
        "unclear_count": row.unclear_count,
        "turn_count": row.turn_count,
    }


# ---- Cart -----------------------------------------------------------------

def _get_or_create_cart(db, session_id: str) -> Cart:
    cart = db.scalar(select(Cart).where(Cart.session_id == session_id))
    if cart is None:
        cart = Cart(session_id=session_id)
        db.add(cart)
        db.flush()
    return cart


def get_cart_items(session_id: str) -> list[dict]:
    with SessionLocal() as db:
        cart = db.scalar(select(Cart).where(Cart.session_id == session_id))
        if cart is None:
            return []
        return [
            {
                "item": ci.menu_item.name,
                "quantity": ci.quantity,
                "price": float(ci.menu_item.price),
            }
            for ci in cart.items
        ]


def add_cart_item(session_id: str, item_name: str, quantity: int) -> None:
    """Merges quantity into an existing cart line instead of replacing it
    (the uq_cart_item constraint on (cart_id, menu_item_id) is what makes
    this safe under concurrent writes - the upsert below either creates
    the row or atomically bumps its quantity)."""
    with SessionLocal() as db:
        menu_item = db.scalar(select(MenuItem).where(MenuItem.name == item_name))
        if menu_item is None:
            raise ValueError(f"Unknown menu item: {item_name}")
        cart = _get_or_create_cart(db, session_id)

        stmt = pg_insert(CartItem).values(
            cart_id=cart.id, menu_item_id=menu_item.id, quantity=quantity
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_cart_item",
            set_={"quantity": CartItem.quantity + quantity},
        )
        db.execute(stmt)
        db.commit()


def remove_cart_item(session_id: str, item_name: str, quantity: int) -> str:
    """Returns a short status string describing what happened, for the
    caller to turn into a customer-facing message: "not_in_cart",
    "insufficient" (asked for more than in cart, nothing changed),
    "removed" (line deleted), or "updated" (quantity reduced)."""
    with SessionLocal() as db:
        cart = db.scalar(select(Cart).where(Cart.session_id == session_id))
        menu_item = db.scalar(select(MenuItem).where(MenuItem.name == item_name))
        if cart is None or menu_item is None:
            return "not_in_cart"

        cart_item = db.scalar(
            select(CartItem).where(
                CartItem.cart_id == cart.id, CartItem.menu_item_id == menu_item.id
            )
        )
        if cart_item is None:
            return "not_in_cart"

        remaining = cart_item.quantity - quantity
        if remaining < 0:
            return "insufficient"
        if remaining == 0:
            db.delete(cart_item)
            db.commit()
            return "removed"
        cart_item.quantity = remaining
        db.commit()
        return "updated"


def get_cart_item_quantity(session_id: str, item_name: str) -> int:
    with SessionLocal() as db:
        cart = db.scalar(select(Cart).where(Cart.session_id == session_id))
        if cart is None:
            return 0
        menu_item = db.scalar(select(MenuItem).where(MenuItem.name == item_name))
        if menu_item is None:
            return 0
        cart_item = db.scalar(
            select(CartItem).where(
                CartItem.cart_id == cart.id, CartItem.menu_item_id == menu_item.id
            )
        )
        return cart_item.quantity if cart_item else 0


def clear_cart(session_id: str) -> None:
    with SessionLocal() as db:
        cart = db.scalar(select(Cart).where(Cart.session_id == session_id))
        if cart is None:
            return
        for item in list(cart.items):
            db.delete(item)
        db.commit()


# ---- Orders -----------------------------------------------------------

def create_order_from_cart(session_id: str) -> dict | None:
    """Snapshots the current cart into a new order (with price_at_order
    frozen at today's menu price) and empties the cart, all in one
    transaction. Returns None if the cart was empty."""
    with SessionLocal() as db:
        cart = db.scalar(select(Cart).where(Cart.session_id == session_id))
        if cart is None or not cart.items:
            return None

        order = Order(session_id=session_id, status="ordered")
        db.add(order)
        db.flush()

        items_snapshot = []
        for cart_item in cart.items:
            order_item = OrderItem(
                order_id=order.id,
                menu_item_id=cart_item.menu_item_id,
                quantity=cart_item.quantity,
                price_at_order=cart_item.menu_item.price,
            )
            db.add(order_item)
            items_snapshot.append(
                {
                    "item": cart_item.menu_item.name,
                    "quantity": cart_item.quantity,
                    "price": float(cart_item.menu_item.price),
                }
            )
            db.delete(cart_item)

        db.commit()
        return {"order_id": order.id, "status": order.status, "items": items_snapshot}


def get_order(order_id: int) -> dict | None:
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None:
            return None
        return {
            "order_id": order.id,
            "status": order.status,
            "items": [
                {
                    "item": oi.menu_item.name,
                    "quantity": oi.quantity,
                    "price": float(oi.price_at_order),
                }
                for oi in order.items
            ],
        }


def update_order_status(order_id: int, status: str) -> dict | None:
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None:
            return None
        order.status = status
        order.updated_at = datetime.now(timezone.utc)
        db.commit()
        return get_order(order_id)
