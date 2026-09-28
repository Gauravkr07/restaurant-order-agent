from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Optional, List, Literal
from pydantic import BaseModel
import os

from helpers.constants import MENU
from helpers.llm import llm


class RequestedItem(BaseModel):
    item: str
    quantity: int


class RequestedItems(BaseModel):
    items: list[RequestedItem]


class Cart(BaseModel):
    item: str
    quantity: int
    price: float


class Order(BaseModel):
    status: str
    order_id: int
    items: list[Cart]


class State(MessagesState):
    query_count: int = 0
    cart: List[Cart]
    order: Order | None = None
    payment: dict[str, any]


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
    return {'query_count': query_c, "messages": [response]}


def route_intent(
    state: State,
) -> Literal[
    "SHOW_MENU",
    "ADD_ITEM",
    "REMOVE_ITEM",
    "VIEW_CART",
    "CHECKOUT",
]:
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
    }
    if intent not in allowed_intents:
        return "SHOW_MENU"
    return intent


def add_item(state: State) -> dict:
    cart_item = state["messages"][-1].content
    menu_text = "\n".join(
        [f"{item}: price={details['price']}, stock={details['stock']}"
         for item, details in MENU.items()])
    messages = [
        ("system",
         f"""
You are a restaurant assistant.Available Menu:{menu_text},
Tasks:
Extract food items and quantities.
Verify the item exists.
Verify stock availability.
Return structured output only."""),
        (
            "human",
            f"User wants: {cart_item}")
    ]
    extractor = llm.with_structured_output(RequestedItems)
    response = extractor.invoke(messages)
    updated_cart = []
    for item in response.items:
        if item.item not in MENU:
            print(f"{item.item}: not available")
            continue
        stock = MENU.get(item.item).get('stock')
        if stock == 0:
            print(f"{item.item}: Out of stock")
            continue
        if item.quantity > stock:
            print(f"{item.item}: Only {stock} available")
            continue
        price = MENU.get(item.item).get("price")
        cart_obj = Cart(
            item=item.item,
            quantity=item.quantity,
            price=price,
        )
        updated_cart.append(cart_obj)
    return {"cart": updated_cart}


def remove_item(state: State):
    user_input = state["messages"][-1].content
    messages = [("system",
                 f"""
You are a restaurant assistant.,
Tasks:
Extract food items and quantities.
Return structured output only.
"""),
                ('human', f"remove item{user_input}")]
    extractor = llm.with_structured_output(RequestedItems)
    response = extractor.invoke(messages)
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
        "messages": result_messages
    }


builder = StateGraph(State)
builder.add_node("show_menu", show_menu)
builder.add_conditional_edges(
    START,
    route_intent,
    {
        "SHOW_MENU": "show_menu",
    }
)
builder.add_edge("show_menu", END)
graph = builder.compile()

from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    test_state = {
        "messages": [
            HumanMessage(content="I want remove 1 pizzas and 1 coke")
        ],
        "cart": [
            Cart(item="pizza", quantity=5, price=200),
            Cart(item="coke", quantity=2, price=50),
        ],
    }
    print(remove_item(test_state))
