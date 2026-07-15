"""
Hoku Health Care - AI Chatbot API Endpoints (Day 3).

FastAPI router exposing the AI chatbot and chat history services.
All endpoints require authentication and persist conversation history
via the CRUD layer.
"""

import logging
import time
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.core.exceptions import UserNotFoundException
from app.crud.chat import (
    get_chat_history_by_user,
    user_exists,
)
from app.schemas.chat import (
    ChatHistoryItem,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
)
from app.services.ai_service import process_chat
from app.utils.validators import sanitize_message, validate_message_length

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

    Steps:
    1. Validate user identity and existence.
    2. Sanitize and validate the incoming message.
    3. Generate AI response via Groq LLM with conversation memory (async, 3.5s timeout).
    4. Persist the conversation turn via the CRUD layer.
    5. Return the response with clinical metadata.

    Args:
        request: Validated chat message payload.
        db: SQLAlchemy database session.
        current_user: Authenticated user dictionary from JWT.

    Returns:
        ChatMessageResponse: AI reply with clinical metadata.

    Raises:
        UserNotFoundException: If the user does not exist.
        HTTPException: If validation fails or an unexpected error occurs.
    """
    request_start = time.perf_counter()

    try:
        # Verify the requesting user matches the authenticated token
        if request.userId != current_user["id"]:
            raise UserNotFoundException(
                detail="User not found or access denied"
            )

        # Verify user exists in the database (placeholder until User model is ready)
        if not user_exists(db, request.userId):
            raise UserNotFoundException()

        # Sanitize and validate input
        clean_message = sanitize_message(request.message)
        if not clean_message or not validate_message_length(clean_message):
            logger.warning(
                "Invalid message from user_id=%s (length=%d)",
                request.userId,
                len(clean_message) if clean_message else 0,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message cannot be empty or exceeds maximum length.",
            )

        # ------------------------------------------------------------------
        # Day 3: Real AI response via Groq + LangChain with conversation memory
        # ------------------------------------------------------------------
        # process_chat now delegates to HokuChatbot which loads per-user
        # memory from the database before calling Groq.
        result = await process_chat(
            message=clean_message,
            user_id=request.userId,
            db=db,
        )

        total_elapsed = time.perf_counter() - request_start
        logger.info(
            "POST /api/ai/chat completed for user_id=%s in %.3fs",
            request.userId,
            total_elapsed,
        )

        # Alert if we breached the 4s NFR (should never happen due to 3.5s hard timeout)
        if total_elapsed > 4.0:
            logger.warning(
                "NFR-02 BREACH: Request for user_id=%s took %.3fs (limit: 4s)",
                request.userId,
                total_elapsed,
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
    "/chat/history",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Chat History",
    description="Retrieve paginated chat history for the authenticated user.",
)
async def get_chat_history(
    limit: int = Query(20, ge=1, le=100, description="Maximum messages to return"),
    skip: int = Query(0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatSessionResponse:
    """
    Retrieve chat history for the authenticated user.

    Converts database rows into a chronological list of human/AI message
    pairs suitable for frontend rendering.

    Args:
        limit: Maximum number of history rows to fetch.
        skip: Number of rows to skip (pagination).
        db: SQLAlchemy database session.
        current_user: Authenticated user dictionary from JWT.

    Returns:
        ChatSessionResponse: User's chat session in chronological order.

    Raises:
        UserNotFoundException: If the user does not exist.
        HTTPException: On unexpected database errors.
    """
    try:
        user_id: int = current_user["id"]

        if not user_exists(db, user_id):
            raise UserNotFoundException()

        # Fetch history (newest first from DB)
        history = get_chat_history_by_user(
            db,
            user_id=user_id,
            limit=limit,
            skip=skip,
        )

        # Flatten rows into chronological human/AI message pairs
        messages: List[ChatHistoryItem] = []
        for entry in reversed(history):
            messages.append(
                ChatHistoryItem(
                    role="human",
                    content=entry.message,
                    timestamp=entry.created_at,
                )
            )
            if entry.ai_response:
                messages.append(
                    ChatHistoryItem(
                        role="ai",
                        content=entry.ai_response,
                        timestamp=entry.created_at,
                    )
                )

        logger.info(
            "Returned %d messages for user_id=%s (limit=%d, skip=%d)",
            len(messages),
            user_id,
            limit,
            skip,
        )

        return ChatSessionResponse(user_id=user_id, messages=messages)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Error retrieving chat history for user %s: %s",
            user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history.",
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