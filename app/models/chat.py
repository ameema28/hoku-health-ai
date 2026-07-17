"""
Hoku Health Care - Chat History Model (SQLAlchemy 2.0).

Declarative model for persisting AI chatbot interactions using
SQLAlchemy 2.0 mapped_column syntax for type safety and modern ORM
patterns.

Day 4 update:
- Added index on intent column for fast analytics queries
  (e.g., "how many emergency intents today?")
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class ChatHistory(Base):
    """
    Represents a single turn in an AI chatbot conversation.

    Each row stores the user message, the AI reply, the classified intent,
    and a timestamp for audit and context retrieval.

    Attributes:
        id: Primary key (auto-increment).
        user_id: Foreign key referencing the users table.
        message: Raw user input (sanitized before persistence).
        ai_response: Generated AI response text.
        intent: Classified intent (e.g., 'symptom', 'booking', 'general').
        created_at: UTC timestamp of the conversation turn.
    """

    __tablename__ = "chat_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    ai_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Composite indexes for fast lookups by user and time-range queries
    # Day 4: Added intent index for analytics (e.g., emergency count by day)
    __table_args__ = (
        Index("ix_chat_history_user_id", "user_id"),
        Index("ix_chat_history_created_at", "created_at"),
        Index("ix_chat_history_intent", "intent"),  # Day 4: Analytics queries
    )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"<ChatHistory(id={self.id}, user_id={self.user_id}, "
            f"intent='{self.intent}', created_at='{self.created_at}')>"
        )

    def to_dict(self) -> dict:
        """
        Serialize the chat history entry to a plain dictionary.

        Returns:
            dict: Flat dictionary with all column values.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "message": self.message,
            "ai_response": self.ai_response,
            "intent": self.intent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }