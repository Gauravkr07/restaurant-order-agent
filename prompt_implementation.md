You have to act like a senior developer, who has expertise in langgraph and agents.

Programming language: python3
Framework: Langgraph

Information:
1. Models and state are defined in models.py
2. All nodes and edges are declared in agent.py
3. I used state query_count, cart (using pydantic models), order (using pydantic models), payment (not implemented yet)
4. I already created basic structure where you can find multiple nodes (route_intent, show_menu). I also added conditional_edge for all nodes. I implemented that user sees menu using show_menu, user gives order though message, and we need to verify that quantity is present from menu. If present, we will add in cart (add_item) which is state. I also implemented remove quantity (remove_item) node and validate through user input, and also can verify cart using 'view_cart' node.
It can create order also if available items are in cart.
5. Session state (query_count, cook_retry_count, status, cart, order) should live server-side, not be round-tripped by the client every time — client should only ever send the message and a session_id.
6. Add exception handling for Groq errors (token expiration, invalid API key, and other Groq errors) — don't let a raw 500 leak out.
7. Provide a simple chat UI.
8. Cooking duration should be a fixed 2 minutes, not random up to 30 for now.
9. The response should always surface order_id at the top level too, not just nested inside order.
10. If cooking fails and retries, the next customer message must resume cooking, not get misrouted into a random intent.
11. Session persistence must survive server restarts — use Redis instead of an in-memory dict, one Redis key per session, JSON-encoded, with TTL-based expiry so abandoned sessions clean themselves up.
12. Routing should be more agentic — don't force ambiguous or off-topic messages into one of the fixed intents. Add an UNCLEAR intent and a clarify node that asks the customer what they meant instead of guessing.
13. If a customer sends 3 consecutive off-topic/unclear messages, close the session with an apologetic message (reuse status = "regretted") and reject further messages to that session (409) until they start a new one.

Task:
1. Append attributes in state: status
2. I want to start from two option, either user can check_menu or either it will be order directly. User can select any of this but user go for verification that user product and their quantity. If all good then we will proceed with add to cart, but if partially available we will check with user want to continue with left item or quantity. If not we will drop, and he will miss one chance. It will check three time (query_count) only, and if not then will return regret message.
3. add_item, node already present, it will add list of dicts into cart. There will be one condition, if cart item already present and user want to add another quantity or item, it will update not replace previous.
4. remove_item and view_cart nodes can be used if user ask to view cart or want to update cart.
5. We will create order using all cart item, and also store in order attribute in state. Order also update the status 'order' and 'order_id' can fetch order details and also items with quantity.
6. It will call cooking_stage node, food preparation started and it will update state 'cooking' and also return duration randomly upto 30 min. If any issue come in cooking and fail, it will retry two time and still not fixed then go back to user and told about refund, also add some sorry message.
7. If cooking done, delivery_item will need to trigger with order_id, items and also change status to delivered.
8. Add proper test coverage — HIGH priority. Testcases should be tracked in prompt_testcase.md, and the README should reference how to run them (see "Testing" section in README.md).
9. Implement Docker so the whole thing (app + Redis) can be run with one command — see Dockerfile / docker-compose.yml and the "Run with Docker" section in README.md.

Careful:
1. Always act kind and polite with customer.
2. Also go stepwise, and try to achieve target.
3. Use GroqAI, as llama and gpt models only.
4. Ask anything if you stuck or confused.
5. Don't hallucinate.

Output:
1. Please also take care about response will be well organized, use pydantic request and response schema if required.

---

## Instructions (dated)

New instructions given from here on, appended as they come in.

- **2026-09-30**: Track every new instruction here going forward, tagged with the date it was given.
- **2026-09-30**: I want to debug langgraph properly, so I can track errors, workflow, latency, and token utilization. i want to implement LangSmith for this — need tracing set up for the graph so I can see each node run, how long it takes, and how many tokens it used.
- **2026-09-30**: Add guardrails to the project: invalid quantity, unknown menu item, negative quantity, prompt injection attempt, tool input validation, maximum workflow iterations, LLM timeout, LLM failure fallback.
- **2026-09-30**: Implement proper logging across the app (currently just a stray print() in show_menu).
- **2026-09-30**: Implement CI/CD for the project.
- **2026-09-30**: Redis should no longer be the permanent source of truth. Move to PostgreSQL + SQLAlchemy + Alembic as the real data store (originally proposed: users, sessions, restaurants, menu_items, carts, cart_items, orders, order_items, payments, agent_runs — with order_id/session_id/user_id linking them, transaction handling, soft-delete where appropriate, indexes, unique constraints). Architecture becomes FastAPI -> LangGraph -> Tools -> PostgreSQL. Redis stays around but only for cache, session/checkpoints, locks, rate limiting, temporary state.
  - Scoped down after review to 6 tables that solve a real current problem: menu_items, sessions, carts, cart_items, orders, order_items. Skipped for now: restaurants , users (no real auth exists, would just be an empty id column), payments (feature not implemented yet), agent_runs (LangSmith tracing + JSON logging already cover this need). These can be added later cheaply if those features become real.

---

## Implementing Agentic AI

- **2026-09-30**: Phase 2 — convert workflow nodes into real tools (search_menu, get_menu_item, add_to_cart, remove_from_cart, get_cart, checkout, cancel_order, get_order_status) so the LLM calls them directly instead of a hardcoded intent -> node mapping. Also asked about sync + parallel execution (asyncio.gather over check_menu/check_inventory/calculate_price/delivery_estimate, async FastAPI/DB/Redis, 3s timeout + retry + fallback on external calls) — reviewed and decided to skip for now: today's app has no independent slow operations to parallelize (menu/inventory are the same DB read, pricing is arithmetic, delivery estimate doesn't exist as a real external call yet). Revisit if a genuinely slow, independent external operation gets added later. Proceeding with tool-based agent only, phased: (1) define tools wrapping db/repository.py, (2) one real tool-calling node as a proof of concept, (3) expand to the rest of the graph, (4) update tests/docs.
  - Phase 1 done: `agent_tools.py` — 8 LangChain tools wrapping db/repository.py, verified against real Postgres. Added `repository.cancel_order` (didn't exist before).
  - Phase 2 done: `verify_items` (in agent.py) now runs `run_add_item_agent` — the LLM is given `[search_menu, get_menu_item, add_to_cart]` via `bind_tools` and decides itself what to check/call in a loop, instead of the old extract-then-check-in-Python-then-add pipeline. The old separate `add_item` node was removed (folded into `verify_items`, since the tools already write to the cart). A second, tool-free structured-output call (`extract_unavailable_items`) turns the loop's final summary into a reliable list of unavailable items with reasons - mixing structured-output requests into the same context as tool-call history was found to trigger a real Groq bug ("Tool choice is required, but model did not call a tool"), so that call is kept in its own clean context. Verified live: full order, and partial-availability (insufficient stock) paths both work correctly against real Postgres + Groq. All 46 tests still passing (conftest's FakeLLM extended with bind_tools support).

