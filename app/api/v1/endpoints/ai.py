"""
Hoku Health Care - AI Chatbot API Endpoints.

FastAPI router exposing the AI chatbot, chat history, RAG debug/seed
endpoints, doctor lookup endpoints, and safety monitoring.

Day 7 additions:
- Enhanced emergency header handling with severity metadata
- Safety monitoring endpoint for metrics
- X-Hoku-Emergency header includes urgency level

Day 8 additions:
- RAG seed endpoint wrapped with timeout to prevent 11s+ delays

Day 10 additions:
- Rate limiting on POST /api/ai/chat, per authenticated user
  (RATE_LIMIT_REQUESTS_PER_MINUTE, default 10) and per client IP
  (CHAT_IP_RATE_PER_MINUTE, default 20), with in-process sliding windows.
  429 responses carry Retry-After and standard RateLimit-* headers.
- Rich OpenAPI request/response examples via ``openapi_extra`` and
  ``responses=`` so /docs is self-documenting for the frontend team.
- Consistent operation ``tags``/``summary``/``operation_id`` metadata.
- Clinical safety unchanged: emergency short-circuit, non-diagnostic
  replies, and the mandatory disclaimer all flow through ``process_chat``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple, Union

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy.orm import Session

from app.ai.emergency_detector import EmergencyDetector
from app.ai.rag import HokuRAG
from app.ai.readiness import check_rag, health_payload
from app.core.client_ip import client_ip
from app.core.config import settings
from app.core.dependencies import get_db
from app.core.security import require_admin, require_patient_or_admin
from app.core.exceptions import UserNotFoundException
from app.core.monitoring import get_metrics
from app.core.rate_limit import (
    RateLimitExceeded,
    get_chat_ip_rate_limiter,
    get_chat_rate_limiter,
)
from app.crud import get_chat_history_by_user, user_exists
from app.crud.crud_doctor import get_doctor_availability, get_doctors_by_specialty
from app.schemas.schemas_chat import (
    ChatHistoryItem,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionResponse,
)
from app.schemas.schemas_doctor import DoctorAvailability, DoctorRead
from app.services.ai_service import process_chat
from app.utils.validators import sanitize_message, validate_message_length

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai")


# ---------------------------------------------------------------------------
# Reusable OpenAPI response envelopes
# ---------------------------------------------------------------------------
_CHAT_RESPONSE_EXAMPLES: Dict[str, Any] = {
    "general": {
        "summary": "General health question",
        "value": {
            "reply": (
                "Staying hydrated and resting usually helps a mild headache. "
                "If it persists or worsens, a General Physician can help. "
                "Please consult a doctor for proper diagnosis."
            ),
            "suggestedSpecialist": "General Physician",
            "severity": "mild",
            "shouldSeeDoctor": False,
            "intent": "general",
            "confidence": 0.88,
            "doctor_suggestion": None,
            "fallback_used": False,
        },
    },
    "symptom_with_doctor": {
        "summary": "Symptom query with doctor suggestion",
        "value": {
            "reply": (
                "A persistent fever with a cough for several days is worth "
                "checking. I can't diagnose the cause, but a General "
                "Physician can evaluate you. Please consult a doctor for "
                "proper diagnosis."
            ),
            "suggestedSpecialist": "General Physician",
            "severity": "moderate",
            "shouldSeeDoctor": True,
            "intent": "symptom",
            "confidence": 0.95,
            "doctor_suggestion": {
                "specialist": "General Physician",
                "doctor_name": "Dr. Sara Ahmed",
                "experience": 12,
                "availability": [
                    {
                        "id": 7,
                        "doctor_id": 3,
                        "day_of_week": 0,
                        "start_time": "09:00",
                        "end_time": "13:00",
                        "is_booked": False,
                    }
                ],
                "doctor_id": 3,
                "requested_specialist": None,
            },
            "fallback_used": False,
        },
    },
    "emergency": {
        "summary": "Emergency escalation (also sets X-Hoku-Emergency headers)",
        "value": {
            **EmergencyDetector.get_urgency_response(EmergencyDetector.URGENCY_HIGH),
            "doctor_suggestion": None,
            "fallback_used": False,
        },
    },
}

_ERROR_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    status.HTTP_400_BAD_REQUEST: {
        "description": "Message is empty after trimming whitespace.",
        "content": {
            "application/json": {
                "example": {
                    "detail": "Message cannot be empty or exceeds maximum length."
                }
            }
        },
    },
    status.HTTP_401_UNAUTHORIZED: {
        "description": "Missing or invalid Bearer token.",
        "content": {"application/json": {"example": {"detail": "Not authenticated"}}},
    },
    status.HTTP_404_NOT_FOUND: {
        "description": "userId does not match the authenticated user.",
        "content": {
            "application/json": {
                "example": {"detail": "User not found or access denied"}
            }
        },
    },
    status.HTTP_429_TOO_MANY_REQUESTS: {
        "description": (
            "Rate limit exceeded: per user (default 10/minute) or per client "
            "IP (default 20/minute). Retry-After says when to retry."
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": "Rate limit exceeded. Try again in 42 seconds.",
                    "correlation_id": "9f2c0d6e4b8a4c1e8d7f3a2b1c0e9d8f",
                }
            }
        },
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "description": "Unexpected server error (no reply was produced).",
        "content": {
            "application/json": {
                "example": {
                    "detail": "An unexpected error occurred while processing your request."
                }
            }
        },
    },
}


# ---------------------------------------------------------------------------
# Day 10: rate-limit dependency
# ---------------------------------------------------------------------------
def _too_many_requests(exc: RateLimitExceeded) -> HTTPException:
    """429 with Retry-After and RateLimit-* headers for a limiter breach."""
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=f"Rate limit exceeded. Try again in {exc.retry_after_seconds} seconds.",
        headers={
            "Retry-After": str(exc.retry_after_seconds),
            "RateLimit-Limit": str(exc.limit),
            "RateLimit-Remaining": "0",
            "RateLimit-Reset": str(exc.retry_after_seconds),
        },
    )


async def enforce_chat_rate_limit(
    request: Request,
    response: Response,
    current_user: Dict[str, Any] = Depends(require_patient_or_admin),
) -> Dict[str, Any]:
    """
    Enforce the chat rate limits: per client IP, then per user.

    Runs as a dependency so the limits are checked before the chat runs.
    The per-IP cap (``CHAT_IP_RATE_PER_MINUTE``, proxy-aware via
    ``client_ip``) is checked first, so many demo sessions from one client
    share one budget; then the per-user limit. On success it attaches the
    per-user ``RateLimit-*`` headers; on either breach it raises 429 with
    ``Retry-After``. Returns the authenticated user so the endpoint can
    reuse it without a second ``get_current_user`` resolution.

    Args:
        request: The inbound request (source of the client IP).
        response: The outbound response, used to attach headers.
        current_user: The authenticated user.

    Returns:
        Dict[str, Any]: The authenticated user record.

    Raises:
        HTTPException: 429 when the IP or the user is over its quota.
    """
    limit = settings.RATE_LIMIT_REQUESTS_PER_MINUTE

    if not settings.RATE_LIMIT_ENABLED:
        # Headers must be present even when enforcement is off (CI contract)
        response.headers["RateLimit-Limit"] = str(limit)
        response.headers["RateLimit-Remaining"] = str(limit)
        return current_user

    ip = client_ip(request)
    try:
        get_chat_ip_rate_limiter().check(ip)
    except RateLimitExceeded as exc:
        # The IP is logged only here, so the proxy setting can be checked
        # against a real client address after a deliberate 429.
        logger.warning(
            "Rate limit exceeded for client IP %s (limit=%d/min)", ip, exc.limit
        )
        raise _too_many_requests(exc) from exc

    user_id = current_user["id"]
    try:
        state = get_chat_rate_limiter().check(str(user_id))
    except RateLimitExceeded as exc:
        logger.warning(
            "Rate limit exceeded for user_id=%s (limit=%d/min)", user_id, exc.limit
        )
        raise _too_many_requests(exc) from exc

    response.headers["RateLimit-Limit"] = str(state.limit)
    response.headers["RateLimit-Remaining"] = str(state.remaining)
    response.headers["RateLimit-Reset"] = str(state.reset_seconds)
    return current_user


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@router.post(
    "/chat",
    response_model=ChatMessageResponse,
    status_code=status.HTTP_200_OK,
    summary="AI Health Chatbot",
    operation_id="chat_with_hoku_ai",
    tags=["AI Chatbot"],
    responses={
        **_ERROR_RESPONSES,
        200: {"description": "AI response with metadata."},
    },
    description=(
        "Send a health question to Hoku AI and receive a non-diagnostic "
        "response, grounded in Hoku Health Care's FAQ knowledge base when relevant.\n\n"
        "- **Rate limits:** per user (default 10/minute) and per client IP "
        "(default 20/minute); 429 with `Retry-After` on breach.\n"
        "- **Emergency detection** runs first (keyword check, then the intent "
        "classifier); emergencies get a fixed urgent reply with "
        "`X-Hoku-Emergency: true`.\n"
        "- Messages over 1000 characters are rejected with 422.\n"
        '- Health replies end with *"Please consult a doctor for proper diagnosis."*; '
        'service, booking and FAQ replies end with *"General information only, '
        'not medical advice."* (the self-harm reply ends with a line about '
        "talking to a doctor or mental-health professional).\n"
        "- `shouldSeeDoctor` is false for booking, service and FAQ replies."
    ),
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "headache": {
                            "summary": "Mild symptom",
                            "value": {
                                "message": "I have a headache and fever for 3 days",
                                "userId": 123,
                            },
                        },
                        "booking": {
                            "summary": "Booking intent",
                            "value": {
                                "message": "How do I book an appointment?",
                                "userId": 123,
                            },
                        },
                        "emergency": {
                            "summary": "Emergency phrasing",
                            "value": {
                                "message": "I have chest pain and can't breathe",
                                "userId": 123,
                            },
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {
                "content": {"application/json": {"examples": _CHAT_RESPONSE_EXAMPLES}}
            }
        },
    },
)
async def chat(
    request: ChatMessageRequest,
    response: Response,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(enforce_chat_rate_limit),
) -> ChatMessageResponse:
    """
    Process a chat message through the Hoku AI health chatbot.

    Steps:
    1. Enforce the per-user rate limit (dependency, before this body runs).
    2. Validate user identity and existence.
    3. Sanitize and validate the incoming message.
    4. Generate an AI response via Groq LLM with conversation memory,
       intent classification, RAG-grounded FAQ retrieval, doctor
       suggestion (Day 6), and post-LLM safety verification (Day 7).
    5. Persist the conversation turn via the CRUD layer with intent.
    6. Return the response with clinical metadata, intent, and doctor
       suggestion, adding the X-Hoku-Emergency header on escalation.

    Args:
        request: The chat request body (message + userId).
        response: The outbound response (headers are set on it).
        db: Injected database session.
        current_user: Authenticated user, resolved by the rate-limit
            dependency.

    Returns:
        ChatMessageResponse: The validated response payload.

    Raises:
        HTTPException: 400 for invalid input, 404 for identity mismatch,
            429 for rate-limit breach, 500 for unexpected errors.
    """
    request_start = time.perf_counter()
    metrics = get_metrics()
    try:
        if request.userId != current_user["id"]:
            raise UserNotFoundException(detail="User not found or access denied")

        if not user_exists(db, request.userId):
            raise UserNotFoundException()

        # Emergency detection runs on the message exactly as typed; the
        # pipeline gets the normalised text (whitespace and control
        # characters only; nothing is escaped or truncated).
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
        is_emergency = result.get("intent") == "emergency"
        emergency_kind = result.pop("emergency_kind", None)

        # Day 10: label the Prometheus request counter with intent/emergency.
        metrics.increment_request(
            "/api/ai/chat",
            intent=str(result.get("intent", "unknown")),
            emergency_flag=is_emergency,
        )

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

        # Every emergency reply is a fixed urgent reply (keyword or intent
        # signal). Header and body carry the same severity.
        if is_emergency:
            severity = result.get("severity") or "severe"
            result["severity"] = severity
            result["shouldSeeDoctor"] = True
            response.headers["X-Hoku-Emergency"] = "true"
            response.headers["X-Hoku-Emergency-Severity"] = severity
            # "self_harm" (gentler wording on the demo page) or an urgency level.
            response.headers["X-Hoku-Emergency-Kind"] = str(emergency_kind or "high")

        # Defensive: ensure severity is never None before Pydantic validation
        if result.get("severity") is None:
            result["severity"] = "unknown"
        return ChatMessageResponse(**result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in chat endpoint: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request.",
        ) from exc


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
@router.get(
    "/chat/history",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Chat History",
    operation_id="get_chat_history",
    tags=["AI Chatbot"],
    description="Retrieve paginated chat history for the authenticated user, oldest first.",
    responses={
        status.HTTP_401_UNAUTHORIZED: _ERROR_RESPONSES[status.HTTP_401_UNAUTHORIZED],
        status.HTTP_404_NOT_FOUND: _ERROR_RESPONSES[status.HTTP_404_NOT_FOUND],
    },
)
async def get_chat_history(
    limit: int = Query(
        20, ge=1, le=100, description="Maximum messages to return", example=20
    ),
    skip: int = Query(0, ge=0, description="Pagination offset", example=0),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_patient_or_admin),
) -> ChatSessionResponse:
    """
    Retrieve chat history for the authenticated user, chronologically.

    Args:
        limit: Page size (1-100).
        skip: Offset for pagination.
        db: Injected database session.
        current_user: Authenticated user.

    Returns:
        ChatSessionResponse: Ordered human/ai message pairs.

    Raises:
        HTTPException: 404 if the user is unknown, 500 on retrieval error.
    """
    try:
        user_id: int = current_user["id"]
        if not user_exists(db, user_id):
            raise UserNotFoundException()

        history = get_chat_history_by_user(db, user_id=user_id, limit=limit, skip=skip)

        messages: List[ChatHistoryItem] = []
        for entry in reversed(history):
            messages.append(
                ChatHistoryItem(
                    role="human", content=entry.message, timestamp=entry.created_at
                )
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
        logger.error(
            "Error retrieving chat history for user %s: %s",
            current_user.get("id"),
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat history.",
        ) from exc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="AI Service Health Check",
    operation_id="ai_health_check",
    tags=["Health"],
    description=(
        "Unauthenticated probe. Always HTTP 200. `rag_ready` is true when the "
        "startup warm-up found FAQ content (embedding model + FAQ index); "
        "`status` is `degraded` otherwise. Same payload as `GET /health`."
    ),
)
async def health_check() -> Dict[str, Any]:
    """
    Check AI service health.

    Returns:
        Dict[str, Any]: ``{"status": "ok" | "degraded", "service", "rag_ready"}``.
        Cheap (no I/O) and auth-free so orchestrators can probe it.
    """
    return health_payload()


# ---------------------------------------------------------------------------
# Day 5: RAG endpoints (PRESERVED)
# ---------------------------------------------------------------------------
@router.post(
    "/rag/seed",
    status_code=status.HTTP_200_OK,
    summary="Seed Hoku FAQ Vector Store",
    operation_id="seed_faq_vector_store",
    tags=["RAG (admin)"],
    description="Triggers seeding of the Hoku Health Care FAQ knowledge base into pgvector.",
)
async def seed_rag(
    db: Session = Depends(get_db),
    _admin: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Trigger FAQ seeding into the vector store (admin only).

    Idempotent: FAQs whose question is already stored (by content hash)
    are skipped, so repeated calls do not duplicate the collection.

    Args:
        db: Injected database session.

    Returns:
        Dict[str, Any]: Documents added and skipped, and the collection name.

    Raises:
        HTTPException: 500 if seeding fails.
    """
    from app.scripts.seed_faqs import FAQS

    try:

        def _do_seed() -> Tuple[int, int]:
            rag = HokuRAG(db=db)
            rag.create_vector_store()
            result = rag.upsert_faq_documents(FAQS)
            check_rag(rag)  # seeding an empty index makes RAG ready
            return result

        added, skipped = await asyncio.to_thread(_do_seed)

        return {
            "status": "ok",
            "documents_added": added,
            "documents_skipped": skipped,
            "collection": settings.COLLECTION_NAME,
        }
    except Exception as exc:
        logger.error("RAG seeding failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to seed FAQ vector store.",
        ) from exc


@router.get(
    "/rag/search",
    status_code=status.HTTP_200_OK,
    summary="Debug FAQ Similarity Search",
    operation_id="debug_faq_similarity_search",
    tags=["RAG (admin)"],
    description="Runs a raw similarity search against the Hoku FAQ knowledge base (debug/admin use).",
)
async def search_rag(
    q: str = Query(
        ...,
        min_length=1,
        description="Query text to search FAQs for",
        example="What services does Hoku offer?",
    ),
    db: Session = Depends(get_db),
    _admin: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Debug endpoint: run similarity_search directly and return raw results.

    Args:
        q: The query string.
        db: Injected database session.

    Returns:
        Dict[str, Any]: The query echoed back with scored FAQ matches.

    Raises:
        HTTPException: 500 on search failure.
    """
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
        logger.error("RAG debug search failed: %s", type(exc).__name__)
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
    operation_id="list_doctors_by_specialty",
    tags=["Doctors"],
    description="Retrieve available doctors filtered by medical specialty, most experienced first.",
    responses={
        status.HTTP_401_UNAUTHORIZED: _ERROR_RESPONSES[status.HTTP_401_UNAUTHORIZED]
    },
)
async def list_doctors_by_specialty(
    specialty: str = Query(
        ...,
        min_length=1,
        description="Medical specialty (e.g., Cardiologist)",
        example="Cardiologist",
    ),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_patient_or_admin),
) -> List[DoctorRead]:
    """
    List available doctors for a given medical specialty.

    Returns doctors where is_available=True, ordered by experience_years DESC.

    Args:
        specialty: The medical specialty to filter on.
        db: Injected database session.
        current_user: Authenticated user.

    Returns:
        List[DoctorRead]: Matching doctors, possibly empty.

    Raises:
        HTTPException: 500 on lookup error.
    """
    try:
        doctors = get_doctors_by_specialty(db, specialty=specialty)
        if not doctors:
            logger.info("No doctors found for specialty=%s", specialty)
            return []
        return [DoctorRead.model_validate(d) for d in doctors]
    except Exception as exc:
        logger.error(
            "Failed to list doctors for specialty=%s: %s", specialty, type(exc).__name__
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve doctor list.",
        ) from exc


@router.get(
    "/doctors/{doctor_id}/availability",
    response_model=List[DoctorAvailability],
    status_code=status.HTTP_200_OK,
    summary="Get Doctor Availability",
    operation_id="get_doctor_availability",
    tags=["Doctors"],
    description="Retrieve a doctor's weekly schedule and booked slots.",
    responses={
        status.HTTP_401_UNAUTHORIZED: _ERROR_RESPONSES[status.HTTP_401_UNAUTHORIZED]
    },
)
async def get_doctor_schedule(
    doctor_id: int = Path(..., ge=1, description="Doctor ID", example=3),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(require_patient_or_admin),
) -> List[DoctorAvailability]:
    """
    Get the weekly availability schedule for a specific doctor.

    Args:
        doctor_id: The doctor's database ID.
        db: Injected database session.
        current_user: Authenticated user.

    Returns:
        List[DoctorAvailability]: Time slots, possibly empty.

    Raises:
        HTTPException: 500 on lookup error.
    """
    try:
        slots = get_doctor_availability(db, doctor_id=doctor_id)
        if not slots:
            logger.info("No availability slots found for doctor_id=%s", doctor_id)
            return []
        return [DoctorAvailability.model_validate(s) for s in slots]
    except Exception as exc:
        logger.error(
            "Failed to get availability for doctor_id=%s: %s",
            doctor_id,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve doctor availability.",
        ) from exc


# ---------------------------------------------------------------------------
# Day 7: Safety monitoring endpoint (PRESERVED)
# ---------------------------------------------------------------------------
@router.get(
    "/monitoring/metrics",
    status_code=status.HTTP_200_OK,
    summary="Safety & Performance Metrics",
    operation_id="get_safety_metrics",
    tags=["Monitoring"],
    description=(
        "Returns the human-readable safety & performance summary. For "
        "Prometheus scraping use the root `/metrics` endpoint instead."
    ),
    responses={
        status.HTTP_401_UNAUTHORIZED: _ERROR_RESPONSES[status.HTTP_401_UNAUTHORIZED]
    },
)
async def get_monitoring_metrics(
    current_user: Dict[str, Any] = Depends(require_admin),
) -> Dict[str, Any]:
    """
    Retrieve current safety and performance metrics.

    Args:
        current_user: Authenticated user.

    Returns:
        Dict[str, Any]: The metrics summary from ``HokuMetrics``.

    Raises:
        HTTPException: 500 on retrieval error.
    """
    try:
        metrics = get_metrics()
        summary = metrics.get_summary()
        logger.info(
            "Monitoring metrics requested by user_id=%s", current_user.get("id")
        )
        return {"status": "ok", "metrics": summary}
    except Exception as exc:
        logger.error("Failed to retrieve monitoring metrics: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve monitoring metrics.",
        ) from exc
