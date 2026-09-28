"""
Restaurant Agent Service
-------------------------
A thin FastAPI service that exposes the LangGraph restaurant agent
(defined in agent.py) over HTTP. agent.py is imported as-is and is
not modified by this file.

Endpoints:
  GET  /menu                 - list the menu
  POST /chat                 - free-form message, runs through the graph (SHOW_MENU intent only today)
  POST /cart/add             - add item(s) to a cart via natural language
  POST /cart/remove          - remove item(s) from a cart via natural language
  GET  /orders                - list order history (in-memory)
  GET  /orders/{order_id}     - get a single order's status
"""
from itertools import count
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

import agent

app = FastAPI(title="Restaurant Agent Service")

# --- in-memory order history (Order status & history feature) ---
_order_id_counter = count(1)
_orders: dict[int, agent.Order] = {}


class ChatRequest(BaseModel):
    message: str
    cart: list[agent.Cart] = []


class CartActionRequest(BaseModel):
    message: str
    cart: list[agent.Cart] = []


class CheckoutRequest(BaseModel):
    cart: list[agent.Cart]


@app.get("/menu")
def get_menu() -> dict[str, Any]:
    return agent.MENU


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    state = {
        "messages": [HumanMessage(content=req.message)],
        "cart": req.cart,
    }
    result = agent.graph.invoke(state)
    last_message = result["messages"][-1]
    content = getattr(last_message, "content", last_message)
    return {"reply": content, "cart": result.get("cart", req.cart)}


@app.post("/cart/add")
def add_to_cart(req: CartActionRequest) -> dict[str, Any]:
    state = {
        "messages": [HumanMessage(content=req.message)],
        "cart": req.cart,
    }
    result = agent.add_item(state)
    return {"cart": result["cart"]}


@app.post("/cart/remove")
def remove_from_cart(req: CartActionRequest) -> dict[str, Any]:
    state = {
        "messages": [HumanMessage(content=req.message)],
        "cart": req.cart,
    }
    result = agent.remove_item(state)
    return {"cart": result["cart"], "messages": result["messages"]}


@app.post("/orders")
def create_order(req: CheckoutRequest) -> agent.Order:
    """Creates an order from the given cart. No payment processing yet."""
    if not req.cart:
        raise HTTPException(status_code=400, detail="Cart is empty")
    order = agent.Order(
        status="pending",
        order_id=next(_order_id_counter),
        items=req.cart,
    )
    _orders[order.order_id] = order
    return order


@app.get("/orders")
def list_orders() -> list[agent.Order]:
    return list(_orders.values())


@app.get("/orders/{order_id}")
def get_order(order_id: int) -> agent.Order:
    order = _orders.get(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
