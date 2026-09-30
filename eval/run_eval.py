"""
Runs eval/dataset.py's scenarios against the REAL agent graph and REAL
Groq LLM (not the fake LLM tests/ uses - this measures actual model
behavior, not code correctness). Requires a real GROQ_API_KEY and a
reachable Postgres (DATABASE_URL) - see README's "Evaluation" section.

Usage: python -m eval.run_eval

Metrics (see prompt_implementation.md's dated notes for the originally
larger scope this was scaled down from):
  - intent accuracy: did route_intent classify the last message as expected?
  - task completion: did the graph end in the expected status?
  - clarification rate: fraction of "clarification"-category scenarios
    that actually got routed to CLARIFY (not forced into a wrong intent)
  - latency: wall-clock seconds for the full graph run
  - token usage: summed across every LLM call in that run (via a
    callback handler, since a single turn can make several calls -
    classification, tool calls, structured-output extraction)
"""
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

import agent
from db import repository
from eval.dataset import SCENARIOS, Scenario


class TokenCounter(BaseCallbackHandler):
    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_end(self, response, **kwargs):
        for generation_list in response.generations:
            for generation in generation_list:
                usage = getattr(generation.message, "usage_metadata", None)
                if usage:
                    self.input_tokens += usage.get("input_tokens", 0)
                    self.output_tokens += usage.get("output_tokens", 0)


def run_scenario(scenario: Scenario) -> dict:
    session_id = str(uuid.uuid4())
    repository.get_or_create_session(session_id)
    for item, qty in scenario.setup_cart:
        repository.add_cart_item(session_id, item, qty)

    counter = TokenCounter()
    state = {
        "session_id": session_id,
        "status": "browsing",
        "query_count": 0,
        "cook_retry_count": 0,
        "unclear_count": 0,
        "order_id": None,
        "messages": [],
    }

    last_message = scenario.turns[-1]
    actual_intent = None
    if scenario.expected_intent is not None:
        intent_state = {**state, "messages": [HumanMessage(content=last_message)]}
        actual_intent = agent.route_intent(intent_state)

    start = time.monotonic()
    result_state = dict(state)
    for turn in scenario.turns:
        result_state = {**result_state, "messages": [HumanMessage(content=turn)]}
        result_state = agent.graph.invoke(result_state, config={"callbacks": [counter]})
    latency_s = time.monotonic() - start

    actual_status = result_state.get("status")
    reply = result_state["messages"][-1].content if result_state.get("messages") else ""

    return {
        "name": scenario.name,
        "category": scenario.category,
        "expected_intent": scenario.expected_intent,
        "actual_intent": actual_intent,
        "intent_correct": actual_intent == scenario.expected_intent if scenario.expected_intent else None,
        "expected_status": scenario.expected_status,
        "actual_status": actual_status,
        "status_correct": actual_status == scenario.expected_status,
        "latency_s": round(latency_s, 2),
        "input_tokens": counter.input_tokens,
        "output_tokens": counter.output_tokens,
        "reply_preview": (reply[:80] + "...") if len(reply) > 80 else reply,
    }


def run_scenario_safe(scenario: Scenario) -> dict:
    try:
        return run_scenario(scenario)
    except Exception as exc:
        return {
            "name": scenario.name,
            "category": scenario.category,
            "expected_intent": scenario.expected_intent,
            "actual_intent": None,
            "intent_correct": False if scenario.expected_intent else None,
            "expected_status": scenario.expected_status,
            "actual_status": None,
            "status_correct": False,
            "latency_s": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reply_preview": f"ERROR: {type(exc).__name__}: {exc}",
        }


def main():
    results = [run_scenario_safe(s) for s in SCENARIOS]

    intent_checked = [r for r in results if r["intent_correct"] is not None]
    intent_accuracy = sum(r["intent_correct"] for r in intent_checked) / len(intent_checked) if intent_checked else None
    task_completion = sum(r["status_correct"] for r in results) / len(results)

    clarification_scenarios = [r for r in results if r["category"] == "clarification"]
    clarification_rate = (
        sum(r["actual_intent"] == "CLARIFY" for r in clarification_scenarios) / len(clarification_scenarios)
        if clarification_scenarios else None
    )

    latencies = [r["latency_s"] for r in results]
    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_output_tokens = sum(r["output_tokens"] for r in results)

    print(f"{'name':<32} {'intent ok':<10} {'status ok':<10} {'latency':<8} {'tokens'}")
    for r in results:
        intent_mark = "-" if r["intent_correct"] is None else ("YES" if r["intent_correct"] else f"NO ({r['actual_intent']})")
        status_mark = "YES" if r["status_correct"] else f"NO ({r['actual_status']})"
        print(f"{r['name']:<32} {intent_mark:<10} {status_mark:<10} {r['latency_s']:<8} {r['input_tokens']+r['output_tokens']}")

    print()
    print("=== Summary ===")
    print(f"Scenarios run:          {len(results)}")
    print(f"Intent accuracy:        {intent_accuracy:.0%}" if intent_accuracy is not None else "Intent accuracy:        n/a")
    print(f"Task completion rate:   {task_completion:.0%}")
    print(f"Clarification rate:     {clarification_rate:.0%}" if clarification_rate is not None else "Clarification rate:     n/a")
    print(f"Latency (median/p95):   {statistics.median(latencies):.2f}s / {sorted(latencies)[int(len(latencies)*0.95)]:.2f}s")
    print(f"Total tokens:           {total_input_tokens + total_output_tokens} ({total_input_tokens} in / {total_output_tokens} out)")

    Path("eval/last_run.json").write_text(json.dumps(results, indent=2))
    print("\nFull results written to eval/last_run.json")


if __name__ == "__main__":
    main()
