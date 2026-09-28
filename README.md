# restaurant-order-agent

A LangGraph-based restaurant ordering agent, exposed as a FastAPI service.

## Structure

- `agent.py` — the LangGraph agent: intent routing, add/remove cart items. Core logic only, no HTTP.
- `service.py` — FastAPI service layer that wraps `agent.py` and exposes it over HTTP. Also tracks order status/history in memory.
- `helpers/constants.py` — static data (currently the `MENU`).
- `helpers/llm.py` — LLM client setup/connectivity (the `ChatGroq` instance), imported by `agent.py`.
- `helpers/models.py` — all Pydantic/data models (`RequestedItem`, `RequestedItems`, `Cart`, `Order`, `State`), imported by `agent.py`.
- `.env.example` — copy to `.env` and fill in your `GROQ_API_KEY`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your real GROQ_API_KEY
```

## Run

```bash
uvicorn service:app --reload
```

Service runs at http://localhost:8000 (docs at `/docs`).

## Endpoints

- `GET /menu` — list menu items
- `POST /chat` — `{ "message": "...", "cart": [] }` — routed through the agent graph
- `POST /cart/add` — `{ "message": "I want 2 pizzas", "cart": [] }`
- `POST /cart/remove` — `{ "message": "remove 1 pizza", "cart": [...] }`
- `POST /orders` — `{ "cart": [...] }` — create an order (status: pending)
- `GET /orders` — list all orders
- `GET /orders/{order_id}` — get one order

## Roadmap

- Payment integration (not yet implemented)
- Persistent storage (currently in-memory)
