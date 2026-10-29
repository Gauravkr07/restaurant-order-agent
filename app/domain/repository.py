"""
Data-access layer between the agent graph and PostgreSQL. Every function
here opens its own short-lived SQLAlchemy session, does one unit of work,
commits, and closes - callers (app/tools/, app/graph/nodes.py) use these
as plain functions and never see a session or raw SQL.

Returns plain dicts/values (not ORM objects) so callers never accidentally
touch a detached ORM instance after its session has closed.
"""
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from app.core.logging import get_logger
from app.domain.base import SessionLocal
from app.domain.models import Cart, CartItem, MenuItem, Order, OrderItem, Session

logger = get_logger(__name__)


class ConcurrentUpdateError(Exception):
    """Raised when an optimistic-concurrency write loses a race: the
    row's `version` had already moved since the caller read it."""


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

class OutOfStockError(Exception):
    """Raised when an order can't be placed because one of its lines no
    longer has enough stock. Carries the offending lines so the caller
    can tell the customer exactly what ran out."""

    def __init__(self, shortfalls: list[dict]):
        self.shortfalls = shortfalls
        detail = ", ".join(
            f"{s['item']} (wanted {s['wanted']}, {s['available']} left)" for s in shortfalls
        )
        super().__init__(f"insufficient stock: {detail}")


def _reserve_stock(db, cart_items) -> None:
    """Decrement menu stock for every cart line, inside the caller's
    transaction. Raises OutOfStockError (rolling the caller back) if any
    line can't be satisfied.

    The decrement is a single conditional UPDATE per line
    (... SET stock = stock - :qty WHERE id = :id AND stock >= :qty), so
    the check and the write are one atomic operation in the database.
    Reading the stock first and then writing it would leave a window
    where two concurrent checkouts both see the last item available and
    both succeed; here the second UPDATE simply matches zero rows.
    """
    shortfalls = []
    for cart_item in cart_items:
        updated = db.execute(
            update(MenuItem)
            .where(MenuItem.id == cart_item.menu_item_id, MenuItem.stock >= cart_item.quantity)
            .values(stock=MenuItem.stock - cart_item.quantity)
        ).rowcount
        if not updated:
            menu_item = db.get(MenuItem, cart_item.menu_item_id)
            shortfalls.append(
                {
                    "item": menu_item.name if menu_item else str(cart_item.menu_item_id),
                    "wanted": cart_item.quantity,
                    "available": menu_item.stock if menu_item else 0,
                }
            )
    if shortfalls:
        db.rollback()
        logger.warning("order_rejected_out_of_stock", extra={"shortfalls": shortfalls})
        raise OutOfStockError(shortfalls)


def create_order_from_cart(session_id: str) -> dict | None:
    """Snapshots the current cart into a new order (with price_at_order
    frozen at today's menu price) and empties the cart, all in one
    transaction. Returns None if the cart was empty."""
    with SessionLocal() as db:
        cart = db.scalar(select(Cart).where(Cart.session_id == session_id))
        if cart is None or not cart.items:
            return None

        _reserve_stock(db, cart.items)

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


def create_order_idempotent(session_id: str, idempotency_key: str) -> tuple[dict | None, bool]:
    """Like create_order_from_cart, but safe to call more than once with
    the same idempotency_key: the first call creates the order, every
    later call with the same key returns that SAME order instead of
    creating a new one - e.g. a customer's three rapid "PAY" clicks
    produce exactly one order.

    Returns (order_dict_or_None, created) - created is False when an
    existing order was returned instead of a new one being made.

    Two layers of protection, in order:
    1. Look up by idempotency_key first (fast path - covers the common
       case of a genuine retry after the first request already
       succeeded and committed).
    2. `SELECT ... FOR UPDATE` on the cart row before reading/clearing
       it, so two requests that both miss the step-1 lookup (because
       they're running concurrently, before either has committed) can't
       both read the same cart and both try to empty it - the second
       request blocks until the first's transaction commits or rolls
       back, then re-checks.
    If both of those somehow still race (e.g. two different session
    rows racing to insert the same externally-supplied key), the
    UNIQUE constraint on idempotency_key is the final backstop: the
    losing INSERT raises IntegrityError, caught here, falling back to
    the step-1 lookup.
    """
    existing = _get_order_by_idempotency_key(idempotency_key)
    if existing is not None:
        return existing, False

    with SessionLocal() as db:
        cart = db.scalar(
            select(Cart).where(Cart.session_id == session_id).with_for_update()
        )
        if cart is None or not cart.items:
            return None, False

        order = Order(session_id=session_id, status="ordered", idempotency_key=idempotency_key)
        db.add(order)
        try:
            db.flush()
        except IntegrityError:
            db.rollback()
            existing = _get_order_by_idempotency_key(idempotency_key)
            assert existing is not None, "unique violation but no matching order found"
            return existing, False

        _reserve_stock(db, cart.items)

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
        db.refresh(order)
        return _order_to_dict(order), True


def _get_order_by_idempotency_key(idempotency_key: str) -> dict | None:
    with SessionLocal() as db:
        order = db.scalar(select(Order).where(Order.idempotency_key == idempotency_key))
        return _order_to_dict(order) if order else None


def get_order(order_id: int) -> dict | None:
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None:
            return None
        return _order_to_dict(order)


def _order_to_dict(order: Order) -> dict:
    return {
        "order_id": order.id,
        "status": order.status,
        "version": order.version,
        "items": [
            {
                "item": oi.menu_item.name,
                "quantity": oi.quantity,
                "price": float(oi.price_at_order),
            }
            for oi in order.items
        ],
    }


def cancel_order(order_id: int, expected_version: int | None = None) -> dict | None:
    """Only valid before cooking has started - once status is "cooking"
    or later, use the refund path in cooking_stage instead.

    If expected_version is given, this is an optimistic-concurrency
    write: the UPDATE only applies WHERE version = expected_version, so
    if another request already changed the order in between the caller
    reading it and calling this, zero rows match and
    ConcurrentUpdateError is raised instead of silently overwriting
    that other change."""
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None or order.status not in ("ordered",):
            return None
        if expected_version is not None and order.version != expected_version:
            raise ConcurrentUpdateError(
                f"order {order_id} version is {order.version}, expected {expected_version}"
            )
        order.status = "cancelled"
        order.version += 1
        order.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(order)
        return _order_to_dict(order)


def update_order_status(order_id: int, status: str, expected_version: int | None = None) -> dict | None:
    """See cancel_order's docstring for what expected_version does."""
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None:
            return None
        if expected_version is not None and order.version != expected_version:
            raise ConcurrentUpdateError(
                f"order {order_id} version is {order.version}, expected {expected_version}"
            )
        order.status = status
        order.version += 1
        order.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(order)
        return _order_to_dict(order)


# ---- Refunds (V3.3 human-in-the-loop) ---------------------------------
#
# There's no separate payments/refund-amount table yet, so the order
# total (sum of order_items) stands in for "refund amount" - see
# prompt_implementation.md's dated notes for why. If a real payments
# table gets built later, swap get_order_total's source for the actual
# amount charged instead of re-deriving it from line items.

REFUND_APPROVAL_THRESHOLD = 100  # INR; above this, a human must approve


def get_order_total(order_id: int) -> float | None:
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None:
            return None
        return sum(float(oi.price_at_order) * oi.quantity for oi in order.items)


# A customer can only request a refund against food they actually
# received. "ordered" is deliberately NOT here: that status means the
# order was placed but has not been delivered, so refunding it would pay
# out for food the kitchen may still be about to serve. An order that
# hasn't started cooking is cancelled (cancel_order), not refunded.
# Already-"refunded"/"pending_refund_approval"/"refund_rejected"/
# "cancelled" orders are excluded so a refund can't be requested twice.
REFUNDABLE_STATUSES = ("delivered",)

# The kitchen's own path. When cooking fails permanently the order's DB
# status is still "ordered" ("regretted" only ever exists as the
# session/conversation status, never written to orders.status), so
# cooking_stage has to be able to refund an undelivered order - the
# customer paid and will get nothing. This is the ONLY caller allowed to
# do that, which is why it takes an explicit flag rather than widening
# REFUNDABLE_STATUSES for everyone.
KITCHEN_FAILURE_REFUNDABLE_STATUSES = ("ordered", "cooking", "delivered")


def request_refund(order_id: int, *, kitchen_failure: bool = False) -> dict | None:
    """Request a refund for an order.

    By default only a delivered order can be refunded - this is the path
    a customer reaches through chat (the request_refund_for_order tool).

    `kitchen_failure=True` is for cooking_stage's own call when cooking
    has permanently failed, which must also cover an undelivered order.
    Customer-facing callers must never set it.

    Returns None if the order doesn't exist or isn't in a refundable
    status for that caller.

    Small refunds (<= REFUND_APPROVAL_THRESHOLD) are auto-approved
    immediately (status -> "refunded"). Large refunds instead pause at
    status "pending_refund_approval" until a human calls approve_refund
    or reject_refund (see the /orders/{id}/approve-refund and
    /reject-refund endpoints in app/api/routes/orders.py) - LangGraph's
    own graph run already ended by the time this matters, so "pausing"
    here just means the order sits in this status rather than the
    conversation itself being suspended (see prompt_implementation.md
    for why this app uses an application-level pause instead of a
    LangGraph checkpointer)."""
    allowed = (
        KITCHEN_FAILURE_REFUNDABLE_STATUSES if kitchen_failure else REFUNDABLE_STATUSES
    )
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None or order.status not in allowed:
            logger.info(
                "refund_refused",
                extra={
                    "order_id": order_id,
                    "order_status": order.status if order else None,
                    "kitchen_failure": kitchen_failure,
                },
            )
            return None
    total = get_order_total(order_id)
    if total is None:
        return None
    if total > REFUND_APPROVAL_THRESHOLD:
        return update_order_status(order_id, "pending_refund_approval")
    return update_order_status(order_id, "refunded")


def approve_refund(order_id: int) -> dict | None:
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None or order.status != "pending_refund_approval":
            return None
        order.status = "refunded"
        order.version += 1
        order.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(order)
        return _order_to_dict(order)


def reject_refund(order_id: int) -> dict | None:
    with SessionLocal() as db:
        order = db.get(Order, order_id)
        if order is None or order.status != "pending_refund_approval":
            return None
        order.status = "refund_rejected"
        order.version += 1
        order.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(order)
        return _order_to_dict(order)
