"""
Hoku Health Care - AI Service Layer (Day 4).

Orchestrates chatbot interactions, delegates persistence to the CRUD
layer, and handles business logic between API endpoints and AI engines.

Day 4 updates:
- process_chat receives intent from chatbot
- Passes intent to create_chat_history for analytics
- Delegates classify_intent to IntentClassifier
- Passes raw_message for emergency detection to avoid HTML-escaping issues
"""

import logging
from typing import Any, Dict, Optional  # <-- Added Optional here

from sqlalchemy.orm import Session

from app.ai.chatbot import HokuChatbot
from app.ai.intent_classifier import IntentClassifier, IntentEnum
from app.ai.memory import HokuConversationMemory
from app.core.exceptions import DatabaseOperationException
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# Singleton chatbot instance (stateless, safe to reuse across requests)
_chatbot: HokuChatbot = HokuChatbot()


async def classify_intent(message: str) -> str:
    """
    Classify the intent of a user message.

    Day 4: Delegates to IntentClassifier using fast_llm
    (llama-3.1-8b-instant) for low-latency classification.

    Args:
        message: Sanitized user message.

    Returns:
        str: Intent label (e.g., "symptom", "booking", "general").
    """
    classifier = IntentClassifier()
    intent, confidence = await classifier.classify_intent(message)
    logger.debug(
        "classify_intent wrapper: intent=%s, confidence=%.2f",
        intent.value,
        confidence,
    )
    return intent.value


async def generate_response(
    message: str,
    user_id: int,
    db: Session,
    raw_message: Optional[str] = None,  # Day 4: Raw message for emergency detection
) -> Dict[str, Any]:
    """
    Generate AI response via HokuChatbot with conversation memory.

    Args:
        message: Sanitized user message.
        user_id: Authenticated user ID.
        db: SQLAlchemy database session.
        raw_message: Optional raw message for emergency detection.

    Returns:
        Dict[str, Any]: Response payload matching ChatMessageResponse.
    """
    return await _chatbot.get_response(message, user_id, db, raw_message=raw_message)


async def process_chat(
    message: str,
    user_id: int,
    db: Session,
    raw_message: Optional[str] = None,  # Day 4: Raw message for emergency detection
) -> Dict[str, Any]:
    """
    Process a user chat message end-to-end.

    Steps:
    1. Generate AI response via HokuChatbot (includes intent classification
       and emergency detection internally).
    2. Extract intent from response for persistence.
    3. Persist conversation via the CRUD layer with intent metadata.
    4. Ensure safety disclaimer is present in the reply.

    Args:
        message: Sanitized user message.
        user_id: Authenticated user ID.
        db: SQLAlchemy database session.
        raw_message: Optional raw message for emergency detection.
            If provided, emergency detection runs on raw_message to avoid
            HTML-escaping issues from sanitize_message().

    Returns:
        Dict[str, Any]: Response payload matching ChatMessageResponse.
    """
    try:
        # ------------------------------------------------------------------
        # Day 4: Generate response with intent classification embedded
        # HokuChatbot.get_response now handles:
        #   - Emergency detection (on raw_message if provided)
        #   - Intent classification
        #   - Intent-aware prompt augmentation
        #   - Main LLM response generation
        # ------------------------------------------------------------------
        response = await generate_response(message, user_id, db, raw_message=raw_message)

        # Extract intent from response for persistence
        intent = response.get("intent", "general")
        confidence = response.get("confidence", 0.0)

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
            # Day 4: Use classified intent instead of hardcoded "general_health"
            intent=intent,
        )

        logger.info(
            "Chat processed for user_id=%s: intent=%s, confidence=%.2f",
            user_id,
            intent,
            confidence,
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
            "intent": "general",
            "confidence": 0.0,
        }