"""
Graph nodes: each decides what happens for one step of the conversation
and returns a state update.

Nodes stay thin on purpose - tool-calling work lives in app/agents/,
data access in app/domain/repository.py, prompt text in
app/agents/prompts.py. A node's job is the business rule (which status
results, which counter increments), not the mechanics.
"""
import random

from app.agents import prompts
from app.agents.cart_agent import run_add_item_agent, run_remove_item_agent
from app.agents.order_agent import run_order_help_agent
from app.agents.runtime import compose_reply
from app.core.guardrails import looks_like_prompt_injection
from app.core.logging import get_logger
from app.domain import repository
from app.graph.state import State, Status
from app.tools import checkout as checkout_tool

logger = get_logger(__name__)

MAX_VERIFY_ATTEMPTS = 3
MAX_COOKING_RETRIES = 2
MAX_UNCLEAR_ATTEMPTS = 3
COOKING_DURATION_MINUTES = 2
COOKING_FAILURE_RATE = 0.2


def _order_label(order_id: int | None) -> str:
    return f"Order #{order_id if order_id else '?'}"


# --- routing ---

ALLOWED_INTENTS = {
    "SHOW_MENU",
    "ADD_ITEM",
    "REMOVE_ITEM",
    "VIEW_CART",
    "CHECKOUT",
    "ORDER_HELP",
    "UNCLEAR",
}

# Intents whose node name differs from the intent label.
_INTENT_ROUTES = {"ADD_ITEM": "VERIFY_ITEMS", "UNCLEAR": "CLARIFY"}


def route_intent(state: State) -> str:
    if state.get("status") == Status.COOKING_RETRY:
        logger.info(
            "route_intent_decision",
            extra={"intent": None, "route": "COOKING_STAGE", "reason": "resuming_cooking_retry"},
        )
        return "COOKING_STAGE"

    user_message = state["messages"][-1].content
    if looks_like_prompt_injection(user_message):
        logger.warning("prompt_injection_detected", extra={"route": "CLARIFY"})
        return "CLARIFY"

    response = compose_reply(prompts.ROUTE_INTENT, f"user input is {user_message}")
    intent = response.content.strip().upper()

    if intent not in ALLOWED_INTENTS:
        logger.warning(
            "route_intent_unrecognized_response",
            extra={"raw_intent": intent, "route": "CLARIFY"},
        )
        return "CLARIFY"

    route = _INTENT_ROUTES.get(intent, intent)
    logger.info("route_intent_decision", extra={"intent": intent, "route": route})
    return route


def route_after_checkout(state: State) -> str:
    # checkout() only sets order_id on a genuine new order; on an empty
    # cart it returns early with order_id explicitly None, so this
    # correctly ends the turn instead of falling through to cooking_stage
    # for a nonexistent order (a real bug the eval suite caught - see
    # eval/ and prompt_implementation.md).
    return "COOK" if state.get("order_id") is not None else "END"


def route_after_cooking(state: State) -> str:
    return "DELIVER" if state.get("status") == Status.COOKING else "END"


# --- nodes ---

def show_menu(state: State) -> dict:
    logger.info("node_show_menu")
    menu_items = "\n".join(
        f"{item['name'].title()} - ₹{item['price']:.0f}"
        for item in repository.get_active_menu_items()
    )
    response = compose_reply(prompts.SHOW_MENU, f"Menu:\n{menu_items}")
    return {
        "query_count": state.get("query_count", 0) + 1,
        "unclear_count": 0,
        "messages": [response],
    }


def clarify(state: State) -> dict:
    logger.info("node_clarify")
    user_message = state["messages"][-1].content
    unclear_count = state.get("unclear_count", 0) + 1

    if unclear_count >= MAX_UNCLEAR_ATTEMPTS:
        logger.warning("session_closed_unclear_limit", extra={"unclear_count": unclear_count})
        response = compose_reply(
            prompts.CLARIFY_EXHAUSTED, f"Customer said: {user_message}"
        )
        return {
            "status": Status.REGRETTED,
            "unclear_count": unclear_count,
            "messages": [response],
        }

    response = compose_reply(prompts.CLARIFY, f"Customer said: {user_message}")
    return {"unclear_count": unclear_count, "messages": [response]}


def verify_items(state: State) -> dict:
    """run_add_item_agent lets the LLM decide which menu/cart tools to
    call. Whatever it manages to add is already committed to the cart in
    Postgres by the time this returns - this function only decides the
    resulting conversation status (cart / verifying / regretted) from the
    verdict."""
    logger.info("node_verify_items")
    verdict = run_add_item_agent(state["session_id"], state["messages"][-1].content)
    added_text = ", ".join(f"{i.quantity} x {i.item}" for i in verdict.added)

    if not verdict.unavailable:
        logger.info("add_item_result", extra={"added": [i.item for i in verdict.added]})
        response = compose_reply(
            prompts.CONFIRM_ADDED, f"Added: {added_text or 'nothing new'}"
        )
        return {
            "status": Status.CART,
            "query_count": 0,
            "unclear_count": 0,
            "messages": [response],
        }

    issues = "; ".join(f"{u.item}: {u.reason}" for u in verdict.unavailable)
    query_count = state.get("query_count", 0) + 1

    if query_count >= MAX_VERIFY_ATTEMPTS:
        logger.warning(
            "session_closed_verify_limit",
            extra={"query_count": query_count, "issues": issues},
        )
        response = compose_reply(prompts.VERIFY_EXHAUSTED, f"Unresolved issues: {issues}")
        return {
            "status": Status.REGRETTED,
            "query_count": query_count,
            "unclear_count": 0,
            "messages": [response],
        }

    logger.info("verify_items_partial", extra={"query_count": query_count, "issues": issues})
    response = compose_reply(
        prompts.VERIFY_PARTIAL, f"Added: {added_text or 'none'}. Issues: {issues}"
    )
    return {
        "status": Status.VERIFYING,
        "query_count": query_count,
        "unclear_count": 0,
        "messages": [response],
    }


def remove_item(state: State) -> dict:
    logger.info("node_remove_item")
    summary = run_remove_item_agent(state["session_id"], state["messages"][-1].content)
    return {"unclear_count": 0, "messages": [("ai", summary)]}


def view_cart(state: State) -> dict:
    logger.info("node_view_cart")
    cart = repository.get_cart_items(state["session_id"])
    if not cart:
        response = compose_reply(prompts.CART_EMPTY, "Tell the customer their cart is empty.")
        return {"unclear_count": 0, "messages": [response]}

    cart_text = "\n".join(f"{i['quantity']} x {i['item']} @ ₹{i['price']}" for i in cart)
    total = sum(i["quantity"] * i["price"] for i in cart)
    response = compose_reply(prompts.CART_PRESENT, f"Cart:\n{cart_text}\nTotal: ₹{total}")
    return {"unclear_count": 0, "messages": [response]}


def order_help(state: State) -> dict:
    """Order tracking, refund requests, and cancellation for an EXISTING
    order by number - the customer states the order id themselves, since
    there's no login linking a session to their past order history yet."""
    logger.info("node_order_help")
    summary = run_order_help_agent(state["messages"][-1].content)
    return {"unclear_count": 0, "messages": [("ai", summary)]}


def checkout(state: State) -> dict:
    """Deterministic single action (create an order from the cart) - no
    decision-making needed, so this calls the checkout tool directly
    rather than running a full tool-calling loop for it."""
    logger.info("node_checkout")
    order = checkout_tool.invoke({"session_id": state["session_id"]})

    if order is None:
        response = compose_reply(
            prompts.CHECKOUT_EMPTY,
            "Tell the customer their cart is empty, so there is nothing to order yet.",
        )
        # Explicitly clear order_id rather than leaving it unset: an
        # unset key here would fall back to the carried-over
        # counters["order_id"] from a PREVIOUS completed order in this
        # same session, which route_after_checkout would then wrongly
        # treat as "a new order was just placed" and re-run cooking/
        # delivery for that old order.
        return {"unclear_count": 0, "order_id": None, "messages": [response]}

    logger.info(
        "order_created",
        extra={"order_id": order["order_id"], "item_count": len(order["items"])},
    )
    cart_text = "\n".join(f"{i['quantity']} x {i['item']}" for i in order["items"])
    response = compose_reply(
        prompts.CHECKOUT_PLACED, f"Order #{order['order_id']} placed:\n{cart_text}"
    )
    return {
        "order_id": order["order_id"],
        "status": Status.ORDERED,
        "unclear_count": 0,
        "messages": [response],
    }


def cooking_stage(state: State) -> dict:
    logger.info("node_cooking_stage")
    order_id = state.get("order_id")
    retry_count = state.get("cook_retry_count", 0)
    cooking_failed = random.random() < COOKING_FAILURE_RATE

    if cooking_failed and retry_count < MAX_COOKING_RETRIES:
        logger.warning("cooking_retry", extra={"order_id": order_id, "attempt": retry_count + 1})
        response = compose_reply(
            prompts.COOKING_RETRY, f"{_order_label(order_id)} is being retried."
        )
        return {
            "status": Status.COOKING_RETRY,
            "cooking_duration": None,
            "cook_retry_count": retry_count + 1,
            "messages": [response],
        }

    if cooking_failed:
        logger.error(
            "cooking_failed_permanently",
            extra={"order_id": order_id, "retries_exhausted": retry_count},
        )
        refund_order = repository.request_refund(order_id) if order_id is not None else None
        pending_approval = (
            refund_order is not None and refund_order["status"] == "pending_refund_approval"
        )
        if pending_approval:
            logger.warning("refund_pending_human_approval", extra={"order_id": order_id})
            response = compose_reply(
                prompts.COOKING_FAILED_PENDING_APPROVAL,
                f"{_order_label(order_id)} could not be cooked after retries; "
                "refund is large and needs manual approval.",
            )
        else:
            response = compose_reply(
                prompts.COOKING_FAILED_REFUND,
                f"{_order_label(order_id)} could not be cooked after retries.",
            )
        return {"status": Status.REGRETTED, "cook_retry_count": 0, "messages": [response]}

    logger.info(
        "cooking_started",
        extra={"order_id": order_id, "duration_minutes": COOKING_DURATION_MINUTES},
    )
    response = compose_reply(
        prompts.COOKING_STARTED,
        f"{_order_label(order_id)} is cooking, estimated {COOKING_DURATION_MINUTES} minutes.",
    )
    return {
        "status": Status.COOKING,
        "cooking_duration": COOKING_DURATION_MINUTES,
        "cook_retry_count": 0,
        "messages": [response],
    }


def delivery(state: State) -> dict:
    logger.info("node_delivery")
    order_id = state.get("order_id")
    order = repository.update_order_status(order_id, "delivered") if order_id is not None else None
    logger.info("order_delivered", extra={"order_id": order_id})
    cart_text = "\n".join(f"{i['quantity']} x {i['item']}" for i in (order["items"] if order else []))
    response = compose_reply(
        prompts.DELIVERED, f"{_order_label(order_id)} delivered:\n{cart_text}"
    )
    return {"status": Status.DELIVERED, "messages": [response]}
