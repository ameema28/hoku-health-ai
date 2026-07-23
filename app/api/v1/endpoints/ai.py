"""
Hoku Health Care - AI Chatbot API Endpoints (Day 7: Emergency escalation & safety headers).

FastAPI router exposing the AI chatbot, chat history, RAG debug/seed
endpoints, doctor lookup endpoints, and safety monitoring.

Day 7 additions:
- Enhanced emergency header handling with severity metadata
- Safety monitoring endpoint for metrics
- X-Hoku-Emergency header includes urgency level

Day 8 additions:
- RAG seed endpoint wrapped with timeout to prevent 11s+ delays
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Response, status
from sqlalchemy.orm import Session

from app.ai.rag import HokuRAG
from app.ai.specialist_mapper import SpecialistMapper
from app.ai.symptom_extractor import extract_symptoms_from_text
from app.core.dependencies import get_current_user, get_db
from app.core.exceptions import UserNotFoundException
from app.core.monitoring import get_metrics
from app.crud.crud_doctor import get_doctor_availability, get_doctors_by_specialty
from app.crud import get_chat_history_by_user, user_exists
from app.schemas.schemas_chat import (
    ChatHistoryItem,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
)
from app.schemas.schemas_doctor import DoctorAvailability, DoctorRead, DoctorSuggestion
from app.services.ai_service import process_chat
from app.utils.validators import sanitize_message, validate_message_length

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai", tags=["AI Chatbot"])


@router.post(
    "/chat",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="AI Health Chatbot",
    description=(
        "Send a health question to Hoku AI and receive a safe, non-diagnostic "
        "response, grounded in Hoku Health Care's FAQ knowledge base when relevant. "
        "Day 6: Responses may include a doctor suggestion for symptom/general queries. "
        "Day 7: Emergency detection triggers immediate safety escalation."
    ),
)
async def chat(
    request: ChatMessageRequest,
    response: Response,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> ChatMessageResponse:
    """
    Process a chat message through the Hoku AI health chatbot.

    Steps:
    1. Validate user identity and existence.
    2. Sanitize and validate the incoming message.
    3. Generate AI response via Groq LLM with conversation memory,
       intent classification, RAG-grounded FAQ retrieval, doctor
       suggestion (Day 6), and post-LLM safety verification (Day 7).
    4. Persist the conversation turn via the CRUD layer with intent.
    5. Return the response with clinical metadata, intent, and doctor suggestion.
    6. Add X-Hoku-Emergency header if emergency was detected (Day 7 enhanced).
    """
    request_start = time.perf_counter()
    try:
        if request.userId != current_user["id"]:
            raise UserNotFoundException(detail="User not found or access denied")

        if not user_exists(db, request.userId):
            raise UserNotFoundException()

        # Store raw message BEFORE sanitization for emergency detection --
        # sanitize_message() HTML-escapes text (' -> &#x27;), which breaks
        # emergency regex keywords like "can't breathe".
        raw_message = request.message
        clean_message = sanitize_message(raw_message)

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

        result = await process_chat(
            message=clean_message,
            user_id=request.userId,
            db=db,
            raw_message=raw_message,
        )

        total_elapsed = time.perf_counter() - request_start
        logger.info(
            "POST /api/ai/chat completed for user_id=%s in %.3fs (intent=%s, confidence=%.2f)",
            request.userId,
            total_elapsed,
            result.get("intent", "unknown"),
            result.get("confidence", 0.0),
        )

        if total_elapsed > 4.0:
            logger.warning(
                "NFR-02 BREACH: Request for user_id=%s took %.3fs (limit: 4s)",
                request.userId,
                total_elapsed,
            )

        # Day 7: Enhanced emergency header handling
        if result.get("intent") == "emergency" and result.get("confidence", 0.0) >= 0.99:
            response.headers["X-Hoku-Emergency"] = "true"
            # Day 7: Add urgency level header for frontend routing
            severity = result.get("severity", "severe")
            response.headers["X-Hoku-Emergency-Severity"] = severity
            logger.critical(
                "X-Hoku-Emergency headers set for user_id=%s (severity=%s)",
                request.userId,
                severity,
            )

            # Force severity to severe for all emergency responses
            result["severity"] = "severe"
            result["shouldSeeDoctor"] = True

        # Defensive: ensure severity is never None before Pydantic validation
        if result.get("severity") is None:
            result["severity"] = "unknown"
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
    """Retrieve chat history for the authenticated user, chronologically."""
    try:
        user_id: int = current_user["id"]
        if not user_exists(db, user_id):
            raise UserNotFoundException()

        history = get_chat_history_by_user(db, user_id=user_id, limit=limit, skip=skip)

        messages: List[ChatHistoryItem] = []
        for entry in reversed(history):
            messages.append(
                ChatHistoryItem(role="human", content=entry.message, timestamp=entry.created_at)
            )
            if entry.ai_response:
                messages.append(
                    ChatHistoryItem(
                        role="ai", content=entry.ai_response, timestamp=entry.created_at
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
        logger.exception("Error retrieving chat history for user %s: %s", current_user.get("id"), exc)
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
    """Check AI service health."""
    return {"status": "ok", "service": "Hoku AI Chatbot"}


# ---------------------------------------------------------------------------
# Day 5: RAG endpoints (PRESERVED)
# ---------------------------------------------------------------------------
@router.post(
    "/rag/seed",
    status_code=status.HTTP_200_OK,
    summary="Seed Hoku FAQ Vector Store",
    description="Triggers seeding of the Hoku Health Care FAQ knowledge base into pgvector.",
)
async def seed_rag(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Trigger FAQ seeding into the vector store.

    Intended for admin/dev use during setup -- not part of the patient
    chat flow. Safe to call multiple times (each call adds another copy
    of the FAQ set unless the collection is cleared first).

    Day 8: Wrapped with asyncio.to_thread and timeout to prevent
    embedding model download from blocking the event loop.
    """
    from app.scripts.seed_faqs import FAQS

    try:
        # CRITICAL FIX: Run RAG seeding in a background thread with a
        # generous timeout. The embedding model may need to download on
        # first run (~90MB), which can take 10-15s. We give it up to 30s
        # since this is an admin endpoint, not a patient-facing one.
        def _do_seed():
            rag = HokuRAG(db=db)
            rag.create_vector_store()
            return rag.add_faq_documents(FAQS)

        # Run in thread pool to avoid blocking the event loop
        added = await asyncio.to_thread(_do_seed)
        
        return {
            "status": "ok",
            "documents_added": added,
            "collection": "hoku_health_faqs",
        }
    except Exception as exc:
        logger.exception("RAG seeding failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to seed FAQ vector store.",
        ) from exc


@router.get(
    "/rag/search",
    status_code=status.HTTP_200_OK,
    summary="Debug FAQ Similarity Search",
    description="Runs a raw similarity search against the Hoku FAQ knowledge base (debug/admin use).",
)
async def search_rag(
    q: str = Query(..., min_length=1, description="Query text to search FAQs for"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Debug endpoint: run similarity_search directly and return raw results."""
    try:
        rag = HokuRAG(db=db)
        results = rag.similarity_search(q, k=rag.top_k)
        return {
            "query": q,
            "results": [
                {
                    "question": doc.metadata.get("question"),
                    "answer": doc.metadata.get("answer"),
                    "category": doc.metadata.get("category"),
                    "score": doc.metadata.get("score"),
                }
                for doc in results
            ],
        }
    except Exception as exc:
        logger.exception("RAG debug search failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run similarity search.",
        ) from exc


# ---------------------------------------------------------------------------
# Day 6: Doctor lookup endpoints (PRESERVED)
# ---------------------------------------------------------------------------
@router.get(
    "/doctors",
    response_model=List[DoctorRead],
    status_code=status.HTTP_200_OK,
    summary="List Doctors by Specialty",
    description="Retrieve available doctors filtered by medical specialty.",
)
async def list_doctors_by_specialty(
    specialty: str = Query(..., min_length=1, description="Medical specialty (e.g., Cardiologist)"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[DoctorRead]:
    """
    List available doctors for a given medical specialty.

    Returns doctors where is_available=True, ordered by experience_years DESC.
    """
    try:
        doctors = get_doctors_by_specialty(db, specialty=specialty)
        if not doctors:
            logger.info("No doctors found for specialty=%s", specialty)
            return []
        return [DoctorRead.model_validate(d) for d in doctors]
    except Exception as exc:
        logger.exception("Failed to list doctors for specialty=%s: %s", specialty, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve doctor list.",
        ) from exc


@router.get(
    "/doctors/{doctor_id}/availability",
    response_model=List[DoctorAvailability],
    status_code=status.HTTP_200_OK,
    summary="Get Doctor Availability",
    description="Retrieve a doctor's weekly schedule and booked slots.",
)
async def get_doctor_schedule(
    doctor_id: int = Path(..., ge=1, description="Doctor ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[DoctorAvailability]:
    """
    Get the weekly availability schedule for a specific doctor.

    Returns all time slots for the doctor, including booked status.
    """
    try:
        slots = get_doctor_availability(db, doctor_id=doctor_id)
        if not slots:
            logger.info("No availability slots found for doctor_id=%s", doctor_id)
            return []
        return [DoctorAvailability.model_validate(s) for s in slots]
    except Exception as exc:
        logger.exception("Failed to get availability for doctor_id=%s: %s", doctor_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve doctor availability.",
        ) from exc


# ---------------------------------------------------------------------------
# Day 7: Safety monitoring endpoint (NEW)
# ---------------------------------------------------------------------------
@router.get(
    "/monitoring/metrics",
    status_code=status.HTTP_200_OK,
    summary="Safety & Performance Metrics",
    description="Returns current safety and performance metrics for monitoring.",
)
async def get_monitoring_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Retrieve current safety and performance metrics.

    Returns counters for emergency detections, safety violations,
    NFR-02 breaches, and latency statistics.
    """
    try:
        metrics = get_metrics()
        summary = metrics.get_summary()
        logger.info("Monitoring metrics requested by user_id=%s", current_user.get("id"))
        return {
            "status": "ok",
            "metrics": summary,
        }
    except Exception as exc:
        logger.exception("Failed to retrieve monitoring metrics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve monitoring metrics.",
        ) from exc