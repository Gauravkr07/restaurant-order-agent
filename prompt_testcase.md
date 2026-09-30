You have to act like a senior QA/test engineer with expertise in pytest and LangGraph agent testing.

Programming language: python3
Framework: pytest

Information:
1. The agent graph lives in agent.py, state/models in helpers/models.py, menu in helpers/constants.py, LLM client in helpers/llm.py.
2. The graph is stateful across turns: status, query_count, cook_retry_count, unclear_count, cart, order are persisted server-side in Redis (see service.py /chat) and passed into/out of every graph invocation.
3. Flow under test (current as of the Redis + agentic-routing + 3-strike-close changes):
   - route_intent -> SHOW_MENU | VERIFY_ITEMS (from ADD_ITEM intent) | REMOVE_ITEM | VIEW_CART | CHECKOUT | CLARIFY (from UNCLEAR intent, or an unrecognized LLM response) | COOKING_STAGE (short-circuit when status == "cooking_retry", bypassing intent classification entirely)
   - clarify: asks a clarifying question and increments unclear_count; on the 3rd consecutive unclear message, apologizes, sets status "regretted" (session closes - see service.py's 409 enforcement)
   - every other terminal node (show_menu, verify_items, add_item, view_cart, remove_item, checkout) resets unclear_count to 0, so only *consecutive* unclear messages count
   - verify_items -> add_item (all requested items available) OR back to user (partial/unavailable, retry) OR regret (3rd failed attempt, query_count == 3)
   - add_item merges quantity into an existing cart line instead of replacing it
   - remove_item reduces/removes cart lines, reports items not in cart
   - view_cart shows cart + total, or an empty-cart message
   - checkout builds an Order from the cart with a new order_id
   - cooking_stage: random failure (random.random() < 0.2), retries up to 2 times (cook_retry_count), then regrets with a refund message if still failing; on success sets cooking_duration to a FIXED value (COOKING_DURATION_MINUTES, currently 2 - not random) and status "cooking"
   - route_after_cooking sends "cooking" status to delivery, anything else (cooking_retry, regretted) ends the turn
   - delivery marks the order "delivered"
   - extract_requested_items() retries once on any groq.GroqError before raising (structured-output extraction can flake on Groq)

Task:
1. Write pytest test cases under a tests/ directory, one file per node/concern:
   - test_route_intent.py
   - test_clarify.py
   - test_verify_items.py
   - test_add_item.py
   - test_remove_item.py
   - test_view_cart.py
   - test_checkout_cooking_delivery.py
   - test_extract_requested_items.py
2. Mock helpers.llm.llm (or agent.llm) for every test - never call the real Groq API. Use a fake object exposing .invoke() (returns a message-like object with .content) and .with_structured_output(schema) (returns an object whose .invoke() returns a schema instance, e.g. RequestedItems, or raises a groq.GroqError subclass to simulate flakiness).
3. Cover these cases explicitly:
   a. verify_items: all requested items available -> status "cart", query_count reset to 0, unclear_count reset to 0.
   b. verify_items: partial availability -> status "verifying", query_count incremented by 1, message asks user to continue or drop.
   c. verify_items: 3rd consecutive failed attempt (query_count reaches 3) -> status "regretted", regret message, query_count == 3.
   d. verify_items: item not on menu at all -> treated as unavailable, not a crash.
   e. add_item: adding an item not yet in cart -> new cart line with correct price/quantity.
   f. add_item: adding more quantity of an item already in cart -> quantity is summed, not replaced (this is the important regression case).
   g. add_item: requested quantity exceeds stock -> item silently skipped, cart unchanged.
   h. remove_item: removing partial quantity -> line updated, not removed.
   i. remove_item: removing exact/more than available quantity -> line removed from cart.
   j. remove_item: removing an item not in the cart -> reports "not in cart", no crash, cart unchanged.
   k. view_cart: non-empty cart -> reply includes item lines and a total.
   l. view_cart: empty cart -> reply says cart is empty, no crash on sum/total.
   m. checkout: non-empty cart -> Order created with incrementing order_id, status "ordered", items match cart.
   n. checkout: empty cart -> no Order created, polite message returned instead.
   o. cooking_stage: forced success path (monkeypatch agent.random.random to return >= 0.2) -> status "cooking", cooking_duration == COOKING_DURATION_MINUTES exactly (not a range check - it's fixed now), cook_retry_count reset to 0.
   p. cooking_stage: forced failure path with retry_count < MAX_COOKING_RETRIES -> status "cooking_retry", cook_retry_count incremented, cooking_duration is None.
   q. cooking_stage: forced failure path with retry_count == MAX_COOKING_RETRIES (retries exhausted) -> status "regretted", refund/apology message, cook_retry_count reset to 0.
   r. delivery: order present -> returned order.status == "delivered", reply mentions the order id.
   s. route_intent: message classified as ADD_ITEM by the (mocked) LLM must route to "VERIFY_ITEMS", not "ADD_ITEM" (regression guard for the intent remap).
   t. route_intent: LLM returns UNCLEAR -> routes to "CLARIFY".
   u. route_intent: LLM returns something outside the allowed intent set entirely (garbage/unparseable) -> routes to "CLARIFY" (not "SHOW_MENU" - this changed from the original fallback behavior).
   v. route_intent: state["status"] == "cooking_retry" -> routes straight to "COOKING_STAGE", bypassing the LLM classification call entirely (assert the fake LLM's classify method was NOT called for this case).
   w. clarify: 1st and 2nd unclear message -> status unchanged (not "regretted"), unclear_count incremented, a clarifying question returned.
   x. clarify: 3rd consecutive unclear message (unclear_count reaches MAX_UNCLEAR_ATTEMPTS) -> status "regretted", apology message, unclear_count == 3.
   y. extract_requested_items: fake LLM raises a groq.GroqError on the first call and succeeds on the second -> returns the successful result (retry works).
   z. extract_requested_items: fake LLM raises a groq.GroqError on every call (exceeds EXTRACTION_RETRY_ATTEMPTS) -> raises that error rather than looping forever or returning None.
4. For cooking_stage's randomness, monkeypatch agent.random.random so tests are deterministic - do not rely on statistical flakiness.
5. Use pytest fixtures (in conftest.py) for: a fake LLM double, a sample MENU-backed cart, and a base State dict, so individual tests stay short.
6. Assert on returned dict keys directly (status, cart, query_count, cook_retry_count, unclear_count, order, messages) - do not assert on exact LLM-generated wording, only on structured fields and that a message was returned.

Careful:
1. Do not hit the real Groq API in any test - everything must be mocked/offline.
2. Do not test wording/tone of LLM replies, only structured state transitions.
3. Keep each test focused on one behavior; prefer many small tests over few large ones.
4. Ask if any expected behavior above is ambiguous rather than guessing.
5. Tests must not require a running Redis server or real network access - agent.py's graph/node functions are tested directly (no service.py/HTTP layer in this pass).

Output:
1. Test files under tests/, using pytest conventions (test_*.py, plain assert statements, fixtures in conftest.py where shared).
2. A short conftest.py providing the fake LLM fixture and sample cart/state fixtures.
3. A "Testing" section added to README.md (or a separate TESTING.md referenced from README.md) explaining how to install test deps and run `pytest`.
