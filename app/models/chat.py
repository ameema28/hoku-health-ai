"""
Hoku Health Care - Chat History Model.

SQLAlchemy model for persisting AI chatbot interactions.
Enables conversation history retrieval and audit trails.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text

from app.core.database import Base


class ChatHistory(Base):
    """
    Represents a single turn in an AI chatbot conversation.

    Attributes:
        id: Primary key.
        user_id: Foreign key to the users table.
        message: User's input message.
        ai_response: AI-generated response text.
        intent: Classified intent of the message (e.g., 'symptom', 'booking').
        created_at: Timestamp of the conversation turn.
    """

    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=True)
    intent = Column(String(100), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_chat_history_user_id", "user_id"),
        Index("ix_chat_history_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"<ChatHistory(id={self.id}, user_id={self.user_id}, "
            f"intent='{self.intent}', created_at='{self.created_at}')>"
        )

    def to_dict(self) -> dict:
        """
        Serialize the chat history entry to a dictionary.

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
