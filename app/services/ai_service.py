"""
Hoku Health Care - AI Service Layer.

Orchestrates chatbot interactions, delegates persistence to the CRUD layer,
and handles business logic between API endpoints and AI engines.
"""

import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.ai.chatbot import HokuChatbot
from app.crud.chat import create_chat_history
from app.core.exceptions import DatabaseOperationException
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# Singleton chatbot instance (stateless, safe to reuse across requests)
_chatbot: HokuChatbot = HokuChatbot()


async def process_chat(message: str, user_id: int, db: Session) -> Dict[str, Any]:
    """
    Process a user chat message end-to-end.

    Steps:
    1. Generate AI response via HokuChatbot (Groq LLM via LangChain).
    2. Classify intent (stubbed for Day 2 — will use fast model in production).
    3. Persist conversation via the CRUD layer.
    4. Ensure safety disclaimer is present in the reply.

    Args:
        message: Sanitized user message.
        user_id: Authenticated user ID.
        db: SQLAlchemy database session.

    Returns:
        Dict[str, Any]: Response payload matching ChatMessageResponse.
    """
    try:
        # ------------------------------------------------------------------
        # Day 2: Real AI response via Groq + LangChain
        # ------------------------------------------------------------------
        # HokuChatbot.get_response is async and handles its own timeout,
        # fallback, and error handling. It returns a dict with all fields.
        response = await _chatbot.get_response(message, user_id)

        # Ensure safety disclaimer is present (double-guard)
        reply: str = response.get("reply", "")
        if SAFETY_DISCLAIMER not in reply:
            reply = f"{reply} {SAFETY_DISCLAIMER}"
            response["reply"] = reply

        # ------------------------------------------------------------------
        # Intent classification (stubbed — fast model integration on Day 3)
        # ------------------------------------------------------------------
        intent = "general_health"

        # ------------------------------------------------------------------
        # Persist via CRUD layer for atomic transaction handling
        # ------------------------------------------------------------------
        create_chat_history(
            db=db,
            user_id=user_id,
            message=message,
            ai_response=reply,
            intent=intent,
        )

        return response

    except DatabaseOperationException:
        # Let database exceptions propagate to the endpoint handler
        raise
    except Exception as exc:
        logger.exception(
            "Error processing chat for user %s: %s",
            user_id,
            exc,
        )
        # Graceful fallback — do not persist failed turns
        return {
            "reply": (
                "I'm sorry, I'm having trouble responding right now. "
                f"Please try again later. {SAFETY_DISCLAIMER}"
            ),
            "suggestedSpecialist": None,
            "severity": "unknown",
            "shouldSeeDoctor": True,
        }