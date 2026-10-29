"""
Restaurant Agent Service
------------------------
FastAPI application wiring: static files, routers, exception handlers.
No business logic lives here - see app/api/routes/ for endpoints,
app/graph/ for the agent graph, app/domain/ for data access.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.errors import register_exception_handlers
from app.api.routes import chat, menu, orders

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Restaurant Agent Service")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

register_exception_handlers(app)
app.include_router(chat.router)
app.include_router(menu.router)
app.include_router(orders.router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
