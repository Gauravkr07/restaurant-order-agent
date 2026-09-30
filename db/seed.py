"""
Seeds menu_items with the same starter menu that used to live in
helpers/constants.py's MENU dict (now retired - the table is the source
of truth). Safe to run multiple times: existing items are left alone,
by name.

Usage: python -m db.seed
"""
from sqlalchemy import select

from db.base import SessionLocal
from db.models import MenuItem

STARTER_MENU = {
    "pizza": {"price": 200, "stock": 20},
    "burger": {"price": 150, "stock": 15},
    "coke": {"price": 50, "stock": 50},
    "french fries": {"price": 120, "stock": 25},
    "sandwich": {"price": 100, "stock": 18},
    "pasta": {"price": 220, "stock": 12},
    "fried rice": {"price": 180, "stock": 10},
    "noodles": {"price": 170, "stock": 14},
    "coffee": {"price": 80, "stock": 30},
    "ice cream": {"price": 90, "stock": 22},
}


def seed_menu() -> None:
    with SessionLocal() as db:
        existing_names = set(db.scalars(select(MenuItem.name)).all())
        added = 0
        for name, details in STARTER_MENU.items():
            if name in existing_names:
                continue
            db.add(MenuItem(name=name, price=details["price"], stock=details["stock"]))
            added += 1
        db.commit()
        print(f"Seeded {added} new menu item(s); {len(existing_names)} already present.")


if __name__ == "__main__":
    seed_menu()
