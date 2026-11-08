"""
Hoku Health Care - User model.

Accounts that can log in. ``role`` is checked against ``ROLES`` both by a
CHECK constraint and by the JWT dependency. ``chat_history.user_id`` has no
foreign key to this table (historic rows predate it); demo-user purging
deletes the matching history explicitly.
"""

from datetime import datetime, timezone

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

ROLES = ("patient", "admin")


class User(Base):
    """A login account (patient or admin, optionally a throwaway demo user)."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    #: Stored lower-cased; unique.
    email: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True
    )
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="patient")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("role IN ('patient', 'admin')", name="ck_users_role"),
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, role='{self.role}', is_demo={self.is_demo})>"
