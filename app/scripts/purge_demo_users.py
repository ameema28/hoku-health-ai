"""
Hoku Health Care - Delete demo users older than 24 hours and their chat history.

Usage:
    python -m app.scripts.purge_demo_users
"""

from typing import Callable, Optional

from sqlalchemy.orm import Session

from app.crud.crud_user import purge_demo_users


def main(session_factory: Optional[Callable[[], Session]] = None) -> int:
    if session_factory is None:
        from app.core.database import SessionLocal

        session_factory = SessionLocal
    db = session_factory()
    try:
        removed = purge_demo_users(db)
    finally:
        db.close()
    print(f"Removed {removed} demo users older than 24 hours.")
    return removed


if __name__ == "__main__":
    main()
