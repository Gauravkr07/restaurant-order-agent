You have to act like a senior QA/test engineer with expertise in pytest and LangGraph agent testing.

Programming language: python3
Framework: pytest

Information:
1. The agent graph lives in agent.py, state/models in helpers/models.py, menu in helpers/constants.py, LLM client in helpers/llm.py.
2. The graph is stateful across turns: status, query_count, cook_retry_count, cart, order are passed in and returned on every invocation (see service.py /chat).
3. Flow under test:
   - route_intent -> SHOW_MENU | VERIFY_ITEMS (from ADD_ITEM intent) | REMOVE_ITEM | VIEW_CART | CHECKOUT
   - verify_items -> add_item (all requested items available) OR back to user (partial/unavailable, retry) OR regret (3rd failed attempt)
   - add_item merges quantity into an existing cart line instead of replacing it
   - remove_item reduces/removes cart lines, reports items not in cart
   - view_cart shows cart + total, or an empty-cart message
   - checkout builds an Order from the cart with a new order_id
   - cooking_stage: random failure, retries up to 2 times (cook_retry_count), then regrets with a refund message if still failing; on success sets cooking_duration (5-30) and status "cooking"
   - delivery marks the order "delivered"

Task:
1. Write pytest test cases under a tests/ directory, one file per node/concern (e.g. test_verify_items.py, test_add_item.py, test_remove_item.py, test_checkout_cooking_delivery.py, test_route_intent.py).
2. Mock helpers.llm.llm (or agent.llm) for every test — never call the real Groq API. Use a fake object exposing .invoke() (returns a message with .content) and .with_structured_output(schema) (returns an object whose .invoke() returns a schema instance, e.g. RequestedItems).
3. Cover these cases explicitly:
   a. verify_items: all requested items available -> status "cart", query_count reset to 0.
   b. verify_items: partial availability -> status "verifying", query_count incremented by 1, message asks user to continue or drop.
   c. verify_items: 3rd consecutive failed attempt -> status "regretted", regret message, query_count == 3.
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
   o. cooking_stage: forced success path (mock random to not fail) -> status "cooking", cooking_duration between 5 and 30, cook_retry_count reset to 0.
   p. cooking_stage: forced failure path with retry_count < 2 -> status "cooking_retry", cook_retry_count incremented, cooking_duration is None.
   q. cooking_stage: forced failure path with retry_count == 2 (retries exhausted) -> status "regretted", refund/apology message, cook_retry_count reset to 0.
   r. delivery: order present -> returned order.status == "delivered", reply mentions the order id.
   s. route_intent: message classified as ADD_ITEM by the (mocked) LLM must route to "VERIFY_ITEMS", not "ADD_ITEM" (regression guard for the intent remap).
   t. route_intent: LLM returns something outside the allowed intent set -> defaults to "SHOW_MENU".
4. For cooking_stage's randomness, monkeypatch agent.random.random (and agent.random.randint where relevant) so tests are deterministic - do not rely on statistical flakiness.
5. Use pytest fixtures for: a fake LLM double, a sample MENU-backed cart, and a base State dict, so individual tests stay short.
6. Assert on returned dict keys directly (status, cart, query_count, cook_retry_count, order, messages) - do not assert on exact LLM-generated wording, only on structured fields and that a message was returned.

Careful:
1. Do not hit the real Groq API in any test - everything must be mocked/offline.
2. Do not test wording/tone of LLM replies, only structured state transitions.
3. Keep each test focused on one behavior; prefer many small tests over few large ones.
4. Ask if any expected behavior above is ambiguous rather than guessing.

Output:
1. Test files under tests/, using pytest conventions (test_*.py, plain assert statements, fixtures in conftest.py where shared).
2. A short conftest.py providing the fake LLM fixture and sample cart/state fixtures.
