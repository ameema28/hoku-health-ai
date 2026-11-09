"""
Hoku Health Care - User CRUD.

Emails are normalised (stripped, lower-cased) on write and on lookup.
"""

import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.security import (
    get_password_hash,
    password_policy_error,
    unusable_password_hash,
)
from app.models.models_chat import ChatHistory
from app.models.safety_log import SafetyLog
from app.models.user import ROLES, User

DEMO_USER_TTL = timedelta(hours=24)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.scalar(select(User).where(User.email == normalize_email(email)))


def create_user(db: Session, email: str, password: str, role: str = "patient") -> User:
    """
    Create a real (non-demo) user.

    Raises:
        ValueError: unknown role, password outside the policy, or email taken.
    """
    if role not in ROLES:
        raise ValueError("Role must be one of: " + ", ".join(ROLES))
    problem = password_policy_error(password)
    if problem:
        raise ValueError(problem)
    if get_user_by_email(db, email) is not None:
        raise ValueError("A user with this email already exists.")
    user = User(
        email=normalize_email(email),
        hashed_password=get_password_hash(password),
        role=role,
        is_active=True,
        is_demo=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def set_password(db: Session, user: User, password: str) -> None:
    """Replace a user's password (policy enforced)."""
    problem = password_policy_error(password)
    if problem:
        raise ValueError(problem)
    user.hashed_password = get_password_hash(password)
    db.commit()


def create_demo_user(db: Session) -> User:
    """A throwaway patient account that cannot log in with a password."""
    user = User(
        email=f"demo-{secrets.token_hex(8)}@demo.hoku.invalid",
        hashed_password=unusable_password_hash(),
        role="patient",
        is_active=True,
        is_demo=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def purge_demo_users(db: Session, now: Optional[datetime] = None) -> int:
    """
    Delete demo users older than 24 hours, with their chat history and
    their safety-log events.

    Returns:
        int: number of demo users deleted.
    """
    cutoff = (now or datetime.now(timezone.utc)) - DEMO_USER_TTL
    stale_ids = list(
        db.scalars(
            select(User.id).where(User.is_demo.is_(True), User.created_at < cutoff)
        )
    )
    if stale_ids:
        db.execute(delete(ChatHistory).where(ChatHistory.user_id.in_(stale_ids)))
        db.execute(delete(SafetyLog).where(SafetyLog.user_id.in_(stale_ids)))
        db.execute(delete(User).where(User.id.in_(stale_ids)))
    db.commit()
    return len(stale_ids)
