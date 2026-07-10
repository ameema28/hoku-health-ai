"""
Hoku Health Care - AI Chatbot API Endpoints.

FastAPI router exposing the AI chatbot and related health AI services.
All endpoints require authentication and persist conversation history.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.schemas.chat import ChatMessageRequest, ChatMessageResponse
from app.services.ai_service import process_chat
from app.utils.validators import sanitize_message, validate_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Chatbot"])


@router.post(
    "/chat",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="AI Health Chatbot",
    description="Send a health question to Hoku AI and receive a "
    "safe, non-diagnostic response with optional specialist guidance.",
)
async def chat(
    request: ChatMessageRequest,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatMessageResponse:
    """
    Process a chat message through the Hoku AI health chatbot.

    Args:
        request: Validated chat message payload.
        db: SQLAlchemy database session.
        current_user: Authenticated user dictionary.

    Returns:
        ChatMessageResponse: AI reply with clinical metadata.

    Raises:
        HTTPException: If message validation fails or AI service errors.
    """
    try:
        # Sanitize user input to prevent injection and ensure length constraints
        clean_message = sanitize_message(request.message)
        if not clean_message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty after sanitization.",
            )

        # Validate user ID consistency (request vs. token)
        if not validate_user_id(request.userId):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID.",
            )

        # Process chat through the AI service layer
        result = await process_chat(
            message=clean_message,
            user_id=request.userId,
            db=db,
        )

        return ChatMessageResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unhandled error in chat endpoint: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request.",
        ) from exc


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="AI Service Health Check",
    description="Returns the operational status of the AI chatbot service.",
)
async def health_check() -> Dict[str, str]:
    """
    Check AI service health.

    Returns:
        Dict[str, str]: Status message indicating service is operational.
    """
    return {"status": "ok", "service": "Hoku AI Chatbot"}
