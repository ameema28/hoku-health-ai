"""
Hoku Health Care - AI Service Layer.

Orchestrates chatbot interactions, persists conversation history,
and handles business logic between API endpoints and AI engines.
"""

import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.ai.chatbot import HokuChatbot
from app.models.chat import ChatHistory
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# Singleton chatbot instance (stateless, safe to reuse across requests)
_chatbot: HokuChatbot = HokuChatbot()


async def process_chat(message: str, user_id: int, db: Session) -> Dict[str, Any]:
    """
    Process a user chat message end-to-end.

    Steps:
    1. Generate AI response via HokuChatbot.
    2. Classify intent (stubbed for setup day).
    3. Persist conversation to chat_history table.
    4. Ensure safety disclaimer is present.

    Args:
        message: Sanitized user message.
        user_id: Authenticated user ID.
        db: SQLAlchemy database session.

    Returns:
        Dict[str, Any]: Response payload matching ChatMessageResponse.
    """
    try:
        # Generate AI response
        response = await _chatbot.get_response(message, user_id)

        # Ensure safety disclaimer is present
        reply: str = response.get("reply", "")
        if SAFETY_DISCLAIMER not in reply:
            reply = f"{reply} {SAFETY_DISCLAIMER}"
            response["reply"] = reply

        # Classify intent (stubbed — will use fast model in production)
        intent = "general_health"

        # Persist to database
        history_entry = ChatHistory(
            user_id=user_id,
            message=message,
            ai_response=reply,
            intent=intent,
        )
        db.add(history_entry)
        db.commit()
        db.refresh(history_entry)

        logger.info("Chat persisted: id=%s, user_id=%s", history_entry.id, user_id)

        return response

    except Exception as exc:
        logger.exception("Error processing chat for user %s: %s", user_id, exc)
        # Return a safe fallback without persisting failed turns
        return {
            "reply": (
                "I'm sorry, I'm having trouble responding right now. "
                f"Please try again later. {SAFETY_DISCLAIMER}"
            ),
            "suggestedSpecialist": None,
            "severity": "unknown",
            "shouldSeeDoctor": True,
        }
