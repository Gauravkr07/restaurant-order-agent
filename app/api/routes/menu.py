from fastapi import APIRouter

from app.domain import repository

router = APIRouter(tags=["menu"])


@router.get("/menu")
def get_menu() -> list[dict]:
    return repository.get_active_menu_items()
