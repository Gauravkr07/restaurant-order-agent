"""
Graph wiring only - no business logic. Every node function lives in
app/graph/nodes.py; this module just declares how they connect.
"""
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import State


def build_graph():
    builder = StateGraph(State)

    builder.add_node("show_menu", nodes.show_menu)
    builder.add_node("verify_items", nodes.verify_items)
    builder.add_node("remove_item", nodes.remove_item)
    builder.add_node("view_cart", nodes.view_cart)
    builder.add_node("checkout", nodes.checkout)
    builder.add_node("cooking_stage", nodes.cooking_stage)
    builder.add_node("delivery", nodes.delivery)
    builder.add_node("clarify", nodes.clarify)
    builder.add_node("order_help", nodes.order_help)

    builder.add_conditional_edges(
        START,
        nodes.route_intent,
        {
            "SHOW_MENU": "show_menu",
            "VERIFY_ITEMS": "verify_items",
            "REMOVE_ITEM": "remove_item",
            "VIEW_CART": "view_cart",
            "CHECKOUT": "checkout",
            "COOKING_STAGE": "cooking_stage",
            "ORDER_HELP": "order_help",
            "CLARIFY": "clarify",
        },
    )

    for terminal_node in ("show_menu", "verify_items", "remove_item", "view_cart", "clarify", "order_help"):
        builder.add_edge(terminal_node, END)

    builder.add_conditional_edges(
        "checkout", nodes.route_after_checkout, {"COOK": "cooking_stage", "END": END}
    )
    builder.add_conditional_edges(
        "cooking_stage", nodes.route_after_cooking, {"DELIVER": "delivery", "END": END}
    )
    builder.add_edge("delivery", END)

    return builder.compile()


graph = build_graph()
