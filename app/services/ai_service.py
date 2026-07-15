"""
Hoku Health Care - AI Service Layer (Day 3).

Orchestrates chatbot interactions, delegates persistence to the CRUD layer,
and handles business logic between API endpoints and AI engines.
"""

import logging
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.ai.chatbot import HokuChatbot
from app.ai.memory import HokuConversationMemory
from app.core.exceptions import DatabaseOperationException
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# Singleton chatbot instance (stateless, safe to reuse across requests)
_chatbot: HokuChatbot = HokuChatbot()


async def classify_intent(message: str) -> str:
    """
    Classify the intent of a user message.

    Placeholder for Day 4: Will use fast_llm (llama-3.1-8b-instant) for
    low-latency intent classification before generating the main response.

    Args:
        message: Sanitized user message.

    Returns:
        str: Intent label (e.g., "symptom", "booking", "general_health").
    """
    # TODO Day 4: Use fast_llm for intent classification
    return "general_health"


async def generate_response(message: str, user_id: int, db: Session) -> Dict[str, Any]:
    """
    Generate AI response via HokuChatbot with conversation memory.

    Args:
        message: Sanitized user message.
        user_id: Authenticated user ID.
        db: SQLAlchemy database session.

    Returns:
        Dict[str, Any]: Response payload matching ChatMessageResponse.
    """
    return await _chatbot.get_response(message, user_id, db)


async def process_chat(message: str, user_id: int, db: Session) -> Dict[str, Any]:
    """
    Process a user chat message end-to-end.

    Steps:
    1. Classify intent (stubbed for Day 4).
    2. Generate AI response via HokuChatbot with memory (Groq LLM via LangChain).
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
        # Day 4: Intent classification (placeholder)
        # ------------------------------------------------------------------
        intent = await classify_intent(message)

        # ------------------------------------------------------------------
        # Day 3: Generate response with conversation memory
        # ------------------------------------------------------------------
        response = await generate_response(message, user_id, db)

        # Ensure safety disclaimer is present (double-guard)
        reply: str = response.get("reply", "")
        if SAFETY_DISCLAIMER not in reply:
            reply = f"{reply} {SAFETY_DISCLAIMER}"
            response["reply"] = reply

        # ------------------------------------------------------------------
        # Coordinate memory save after successful response
        # ------------------------------------------------------------------
        memory_manager = HokuConversationMemory()
        memory_manager.save_memory(
            user_id=user_id,
            human_message=message,
            ai_message=reply,
            db=db,
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