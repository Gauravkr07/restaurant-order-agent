"""
Order agent: anything about an EXISTING order by number - checking its
status, requesting a refund, or cancelling it.

Not the current cart/checkout (that's cart_agent and the checkout node).
No session_id is needed here since these tools only take an order_id,
which the customer must state themselves - there's no login linking a
session to a history of past orders yet.
"""
from app.agents import prompts
from app.agents.runtime import run_tool_loop
from app.tools import cancel_order, get_order_status, request_refund_for_order

ORDER_HELP_TOOLS = [get_order_status, request_refund_for_order, cancel_order]


def run_order_help_agent(user_message: str) -> str:
    return run_tool_loop(ORDER_HELP_TOOLS, prompts.ORDER_HELP_AGENT, user_message)
