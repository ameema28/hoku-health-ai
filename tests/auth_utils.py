"""Helpers for tests that need real users and tokens."""

import secrets
from typing import Optional

from sqlalchemy.orm import Session

from app.core.security import create_access_token, unusable_password_hash
from app.models.user import User


def add_user(
    db: Session,
    user_id: Optional[int] = 1,
    role: str = "patient",
    email: Optional[str] = None,
    is_active: bool = True,
    is_demo: bool = False,
    hashed_password: Optional[str] = None,
) -> User:
    """Insert a user row (no password hashing unless a hash is given)."""
    user = User(
        id=user_id,
        email=email or f"user-{secrets.token_hex(6)}@test.hoku.invalid",
        hashed_password=hashed_password or unusable_password_hash(),
        role=role,
        is_active=is_active,
        is_demo=is_demo,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def bearer(user_id: int = 1, role: str = "patient") -> str:
    """``Authorization`` header value with a real signed token."""
    return f"Bearer {create_access_token(user_id, role)}"
