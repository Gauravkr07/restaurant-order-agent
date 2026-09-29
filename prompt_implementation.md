You have to act like a senior developer, who has expertise in langgraph and agents.

Programming language: python3
Framework: Langgraph

Information:
1. Models and state are defined in models.py
2. All nodes and edges are declared in agent.py
3. I used state query_count, cart (using pydantic models), order (using pydantic models), payment (not implemented yet)
4. I already created basic structure where you can find multiple nodes (route_intent, show_menu). I also added conditional_edge for all nodes. I implemented that user sees menu using show_menu, user gives order though message, and we need to verify that quantity is present from menu. If present, we will add in cart (add_item) which is state. I also implemented remove quantity (remove_item) node and validate through user input, and also can verify cart using 'view_cart' node.
It can create order also if available items are in cart.

Task:
1. Append attributes in state: status
2. I want to start from two option, either user can check_menu or either it will be order directly. User can select any of this but user go for verification that user product and their quantity. If all good then we will proceed with add to cart, but if partially available we will check with user want to continue with left item or quantity. If not we will drop, and he will miss one chance. It will check three time (query_count) only, and if not then will return regret message.
3. add_item, node already present, it will add list of dicts into cart. There will be one condition, if cart item already present and user want to add another quantity or item, it will update not replace previous.
4. remove_item and view_cart nodes can be used if user ask to view cart or want to update cart.
5. We will create order using all cart item, and also store in order attribute in state. Order also update the status 'order' and 'order_id' can fetch order details and also items with quantity.
6. It will call cooking_stage node, food preparation started and it will update state 'cooking' and also return duration randomly upto 30 min. If any issue come in cooking and fail, it will retry two time and still not fixed then go back to user and told about refund, also add some sorry message.
7. If cooking done, delivery_item will need to trigger with order_id, items and also change status to delivered.


Careful:
1. Always act kind and polite with customer.
2. Also go stepwise, and try to achieve target.
3. Use GroqAI, as llama and gpt models only.
4. Ask anything if you stuck or confused.
5. Don't hallucinate.

Output:
1. Please also take care about response will be well organized, use pydantic request and response schema if required.
