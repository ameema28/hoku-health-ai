"""
Hoku Health Care - Safety Log Model (Day 7).

SQLAlchemy 2.0 mapped model for persisting clinical safety violations,
emergency escalations, and response audit trails. Supports analytics
queries for safety monitoring and compliance reporting.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, String, Text, Index
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SafetyLog(Base):
    """
    Represents a clinical safety event or response audit entry.

    Attributes:
        id: Primary key (auto-increment).
        user_id: Optional foreign key to users table (nullable for unauthenticated events).
        message: The user message that triggered the safety event.
        ai_response: The AI response that was audited (or emergency response).
        violation_type: Category of safety event.
            - "diagnosis_attempt": AI attempted a definitive diagnosis.
            - "prescription_advice": AI gave prescription/dosage guidance.
            - "missing_disclaimer": Response lacked mandatory disclaimer.
            - "emergency_triggered": Emergency keywords detected in user message.
            - "safety_sanitized": Response was modified by safety guardrails.
        severity: Impact level — "high", "moderate", or "low".
        created_at: UTC timestamp of the event.
    """

    __tablename__ = "safety_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    ai_response: Mapped[str] = mapped_column(Text, nullable=False)
    violation_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Composite indexes for fast analytics queries
    __table_args__ = (
        Index("ix_safety_logs_user_id", "user_id"),
        Index("ix_safety_logs_violation_type", "violation_type"),
        Index("ix_safety_logs_severity", "severity"),
        Index("ix_safety_logs_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<SafetyLog(id={self.id}, user_id={self.user_id}, "
            f"violation_type='{self.violation_type}', severity='{self.severity}', "
            f"created_at='{self.created_at}')>"
        )

    def to_dict(self) -> dict:
        """
        Serialize the safety log entry to a plain dictionary.

        Returns:
            dict: Flat dictionary with all column values.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "message": self.message,
            "ai_response": self.ai_response,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }