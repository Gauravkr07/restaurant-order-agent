import random
from itertools import count

import groq
from langgraph.graph import StateGraph, START, END
from typing import Optional, List, Literal
import os

from helpers.constants import MENU
from helpers.llm import llm
from helpers.models import RequestedItem, RequestedItems, Cart, Order, State

MAX_VERIFY_ATTEMPTS = 3
MAX_COOKING_RETRIES = 2
MAX_UNCLEAR_ATTEMPTS = 3
COOKING_DURATION_MINUTES = 2
EXTRACTION_RETRY_ATTEMPTS = 2
_order_id_counter = count(1)


def extract_requested_items(human_message: str, system_prompt: str) -> RequestedItems:
    """Runs structured-item extraction, retrying once if the model flakes
    (Groq occasionally errors with "Tool choice is required, but model did
    not call a tool" instead of returning structured output)."""
    extractor = llm.with_structured_output(RequestedItems)
    messages = [
        ("system", system_prompt),
        ("human", human_message),
    ]
    last_error: groq.GroqError | None = None
    for _ in range(EXTRACTION_RETRY_ATTEMPTS):
        try:
            return extractor.invoke(messages)
        except groq.GroqError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def show_menu(state: State) -> dict:
    query_c = state.get("query_count", 0) + 1
    menu_items = "\n".join([f"{item.title()} - ₹{details['price']}" for item, details in MENU.items()])
    print("ffff", menu_items)
    messages = [
        ("system", """
You are a resteraunt manager.
please show the menu to customer
Ask then what you would like to order
Be friendly and concise."""),
        (
            "human", f"Menu:\n{menu_items}")
    ]
    response = llm.invoke(messages)
    return {'query_count': query_c, "unclear_count": 0, "messages": [response]}


def route_intent(
    state: State,
) -> Literal[
    "SHOW_MENU",
    "VERIFY_ITEMS",
    "REMOVE_ITEM",
    "VIEW_CART",
    "CHECKOUT",
    "COOKING_STAGE",
    "CLARIFY",
]:
    if state.get("status") == "cooking_retry":
        return "COOKING_STAGE"

    user_message = state["messages"][-1].content
    messages = [
        ("system",
         """Classify the customer's latest message into exactly one intent.
Allowed intents:
SHOW_MENU
ADD_ITEM
REMOVE_ITEM
VIEW_CART
CHECKOUT
UNCLEAR

Use UNCLEAR only if the message genuinely does not map to any of the
other intents (e.g. small talk, a question you can't answer from intent
alone, or something too vague to act on). Do not guess an intent just
to avoid UNCLEAR.

Return only one allowed intent.
Do not add punctuation or explanation."""),
        ("human", f"user input is {user_message}"), ]
    response = llm.invoke(messages)
    intent = response.content.strip().upper()
    allowed_intents = {
        "SHOW_MENU",
        "ADD_ITEM",
        "REMOVE_ITEM",
        "VIEW_CART",
        "CHECKOUT",
        "UNCLEAR",
    }
    if intent not in allowed_intents:
        return "CLARIFY"
    if intent == "UNCLEAR":
        return "CLARIFY"
    if intent == "ADD_ITEM":
        return "VERIFY_ITEMS"
    return intent


def clarify(state: State) -> dict:
    user_message = state["messages"][-1].content
    unclear_c = state.get("unclear_count", 0) + 1

    if unclear_c >= MAX_UNCLEAR_ATTEMPTS:
        response = llm.invoke([
            ("system",
             "You are a polite restaurant assistant. The customer has sent "
             "several messages in a row that are unrelated to ordering food "
             "here. Apologize kindly, explain you're only able to help with "
             "the menu and orders at this restaurant, and let them know "
             "they're welcome to start a new conversation anytime."),
            ("human", f"Customer said: {user_message}"),
        ])
        return {
            "status": "regretted",
            "unclear_count": unclear_c,
            "messages": [response],
        }

    response = llm.invoke([
        ("system",
         "You are a polite restaurant assistant. The customer said something "
         "that doesn't clearly map to an action you can take. Ask a short, "
         "friendly clarifying question, offering the things you can help "
         "with: seeing the menu, ordering, viewing the cart, or checking out. "
         "Don't guess what they meant."),
        ("human", f"Customer said: {user_message}"),
    ])
    return {"unclear_count": unclear_c, "messages": [response]}


def verify_items(state: State) -> dict:
    user_message = state["messages"][-1].content
    menu_text = "\n".join(
        [f"{item}: stock={details['stock']}" for item, details in MENU.items()])
    requested = extract_requested_items(
        f"User wants: {user_message}",
        f"""
You are a restaurant assistant. Available Menu:
{menu_text}
Task:
Extract every food item and quantity the customer is asking for.
Return structured output only.""",
    )

    available: list[RequestedItem] = []
    unavailable: list[str] = []
    for req in requested.items:
        menu_entry = MENU.get(req.item)
        if menu_entry is None:
            unavailable.append(f"{req.item} is not on our menu")
            continue
        stock = menu_entry["stock"]
        if stock == 0:
            unavailable.append(f"{req.item} is out of stock")
            continue
        if req.quantity > stock:
            unavailable.append(
                f"only {stock} {req.item} available (you asked for {req.quantity})")
            continue
        available.append(req)

    if not unavailable:
        return {"status": "cart", "query_count": 0, "unclear_count": 0}

    query_c = state.get("query_count", 0) + 1
    if query_c >= MAX_VERIFY_ATTEMPTS:
        response = llm.invoke([
            ("system",
             "You are a polite restaurant assistant. Apologize that the requested "
             "items could not be fulfilled after multiple attempts and let the "
             "customer know they can start over anytime."),
            ("human", f"Unresolved issues: {'; '.join(unavailable)}"),
        ])
        return {
            "status": "regretted",
            "query_count": query_c,
            "unclear_count": 0,
            "messages": [response],
        }

    available_note = ", ".join(f"{i.quantity} x {i.item}" for i in available) or "none"
    response = llm.invoke([
        ("system",
         "You are a polite restaurant assistant. Some requested items are not "
         "fully available. Tell the customer which items/quantities are available "
         "and which are not, and ask if they'd like to continue with just the "
         "available items/quantities or drop the order."),
        ("human",
         f"Available: {available_note}. Issues: {'; '.join(unavailable)}"),
    ])
    return {
        "status": "verifying",
        "query_count": query_c,
        "unclear_count": 0,
        "messages": [response],
    }


def add_item(state: State) -> dict:
    """Re-verifies the last request against MENU and merges it into the cart.

    Runs after verify_items has already confirmed at least the available
    portion of the order, so this only adds items that pass stock checks
    again here (defends against stock changing between the two calls).
    """
    user_message = state["messages"][-1].content
    menu_text = "\n".join(
        [f"{item}: price={details['price']}, stock={details['stock']}"
         for item, details in MENU.items()])
    response = extract_requested_items(
        f"User wants: {user_message}",
        f"""
You are a restaurant assistant.Available Menu:{menu_text},
Tasks:
Extract food items and quantities.
Verify the item exists.
Verify stock availability.
Return structured output only.""",
    )

    cart_lookup = {item.item: item for item in state.get("cart", [])}
    added: list[RequestedItem] = []
    for item in response.items:
        menu_entry = MENU.get(item.item)
        if menu_entry is None or menu_entry["stock"] < item.quantity:
            continue
        existing = cart_lookup.get(item.item)
        if existing:
            existing.quantity += item.quantity
        else:
            cart_lookup[item.item] = Cart(
                item=item.item,
                quantity=item.quantity,
                price=menu_entry["price"],
            )
        added.append(item)

    updated_cart = list(cart_lookup.values())
    added_text = ", ".join(f"{i.quantity} x {i.item}" for i in added) or "nothing new"
    confirmation = llm.invoke([
        ("system", "You are a polite restaurant assistant. Confirm what was "
                   "just added to the cart and ask if the customer wants "
                   "anything else."),
        ("human", f"Added: {added_text}"),
    ])
    return {
        "cart": updated_cart,
        "status": "cart",
        "unclear_count": 0,
        "messages": [confirmation],
    }


def view_cart(state: State) -> dict:
    cart = state.get("cart", [])
    if not cart:
        response = llm.invoke([
            ("system", "You are a polite restaurant assistant."),
            ("human", "Tell the customer their cart is empty."),
        ])
        return {"unclear_count": 0, "messages": [response]}
    cart_text = "\n".join(
        f"{item.quantity} x {item.item} @ ₹{item.price}" for item in cart)
    total = sum(item.quantity * item.price for item in cart)
    response = llm.invoke([
        ("system", "You are a polite restaurant assistant. Present the "
                   "customer's cart clearly with the total."),
        ("human", f"Cart:\n{cart_text}\nTotal: ₹{total}"),
    ])
    return {"unclear_count": 0, "messages": [response]}


def remove_item(state: State):
    user_input = state["messages"][-1].content
    response = extract_requested_items(
        f"remove item{user_input}",
        """
You are a restaurant assistant.,
Tasks:
Extract food items and quantities.
Return structured output only.
""",
    )
    cart_lookup = {
        item.item: item
        for item in state["cart"]}
    result_messages = []
    for requested_item in response.items:
        cart_item = cart_lookup.get(
            requested_item.item)
        if not cart_item:
            result_messages.append(f"{requested_item.item} is not in cart.")
            continue
        remaining = (
            cart_item.quantity
            - requested_item.quantity
        )
        if remaining < 0:
            result_messages.append(
                f"Only {cart_item.quantity} "
                f"{requested_item.item} available in cart.")
            continue
        if remaining == 0:
            del cart_lookup[
                requested_item.item
            ]
            result_messages.append(
                f"{requested_item.item} removed from cart.")
        else:
            cart_item.quantity = remaining
            result_messages.append(
                f"{requested_item.item} quantity updated to "
                f"{remaining}.")
    updated_cart = list(cart_lookup.values())
    return {
        "cart": updated_cart,
        "unclear_count": 0,
        "messages": result_messages
    }


def checkout(state: State) -> dict:
    cart = state.get("cart", [])
    if not cart:
        response = llm.invoke([
            ("system", "You are a polite restaurant assistant."),
            ("human", "Tell the customer their cart is empty, so there is "
                      "nothing to order yet."),
        ])
        return {"unclear_count": 0, "messages": [response]}

    order = Order(status="ordered", order_id=next(_order_id_counter), items=cart)
    cart_text = "\n".join(f"{i.quantity} x {i.item}" for i in cart)
    response = llm.invoke([
        ("system", "You are a polite restaurant assistant. Confirm the order "
                   "was placed and share the order id."),
        ("human", f"Order #{order.order_id} placed:\n{cart_text}"),
    ])
    return {"order": order, "status": "ordered", "unclear_count": 0, "messages": [response]}


def cooking_stage(state: State) -> dict:
    order = state.get("order")
    retry_count = state.get("cook_retry_count", 0)

    cooking_failed = random.random() < 0.2
    if cooking_failed and retry_count < MAX_COOKING_RETRIES:
        response = llm.invoke([
            ("system", "You are a polite restaurant assistant. Let the "
                       "customer know there was a small hiccup in the kitchen "
                       "and you're retrying their order."),
            ("human", f"Order #{order.order_id if order else '?'} is being retried."),
        ])
        return {
            "status": "cooking_retry",
            "cooking_duration": None,
            "cook_retry_count": retry_count + 1,
            "messages": [response],
        }

    if cooking_failed:
        response = llm.invoke([
            ("system", "You are a polite restaurant assistant. Apologize "
                       "sincerely that the kitchen could not prepare the order "
                       "after multiple attempts, and let the customer know a "
                       "refund will be issued."),
            ("human", f"Order #{order.order_id if order else '?'} could not "
                      "be cooked after retries."),
        ])
        return {
            "status": "regretted",
            "cook_retry_count": 0,
            "messages": [response],
        }

    duration = COOKING_DURATION_MINUTES
    response = llm.invoke([
        ("system", "You are a polite restaurant assistant. Tell the customer "
                   "cooking has started and give the estimated time."),
        ("human", f"Order #{order.order_id if order else '?'} is cooking, "
                  f"estimated {duration} minutes."),
    ])
    return {
        "status": "cooking",
        "cooking_duration": duration,
        "cook_retry_count": 0,
        "messages": [response],
    }


def delivery(state: State) -> dict:
    order = state.get("order")
    if order:
        order = order.model_copy(update={"status": "delivered"})
    cart_text = "\n".join(f"{i.quantity} x {i.item}" for i in (order.items if order else []))
    response = llm.invoke([
        ("system", "You are a polite restaurant assistant. Tell the customer "
                   "their order has been delivered, thank them warmly."),
        ("human", f"Order #{order.order_id if order else '?'} delivered:\n{cart_text}"),
    ])
    return {"order": order, "status": "delivered", "messages": [response]}


def route_after_verify(state: State) -> Literal["ADD_ITEM", "END"]:
    if state.get("status") == "cart":
        return "ADD_ITEM"
    return "END"


def route_after_cooking(state: State) -> Literal["DELIVER", "END"]:
    if state.get("status") == "cooking":
        return "DELIVER"
    return "END"


builder = StateGraph(State)
builder.add_node("show_menu", show_menu)
builder.add_node("verify_items", verify_items)
builder.add_node("add_item", add_item)
builder.add_node("remove_item", remove_item)
builder.add_node("view_cart", view_cart)
builder.add_node("checkout", checkout)
builder.add_node("cooking_stage", cooking_stage)
builder.add_node("delivery", delivery)
builder.add_node("clarify", clarify)

builder.add_conditional_edges(
    START,
    route_intent,
    {
        "SHOW_MENU": "show_menu",
        "VERIFY_ITEMS": "verify_items",
        "REMOVE_ITEM": "remove_item",
        "VIEW_CART": "view_cart",
        "CHECKOUT": "checkout",
        "COOKING_STAGE": "cooking_stage",
        "CLARIFY": "clarify",
    }
)
builder.add_conditional_edges(
    "verify_items",
    route_after_verify,
    {
        "ADD_ITEM": "add_item",
        "END": END,
    }
)
builder.add_edge("show_menu", END)
builder.add_edge("add_item", END)
builder.add_edge("remove_item", END)
builder.add_edge("view_cart", END)
builder.add_edge("clarify", END)
builder.add_edge("checkout", "cooking_stage")
builder.add_conditional_edges(
    "cooking_stage",
    route_after_cooking,
    {
        "DELIVER": "delivery",
        "END": END,
    }
)
builder.add_edge("delivery", END)
graph = builder.compile()




# from langchain_core.messages import HumanMessage

# if __name__ == "__main__":
#     test_state = {
#         "messages": [
#             HumanMessage(content="I want remove 1 pizzas and 1 coke")
#         ],
#         "cart": [
#             Cart(item="pizza", quantity=5, price=200),
#             Cart(item="coke", quantity=2, price=50),
#         ],
#     }
#     print(remove_item(test_state))
