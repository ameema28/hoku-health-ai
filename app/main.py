"""
Hoku Health Care - FastAPI Application Entry Point (Day 10: Production).

Wires together the AI router, observability, and lifecycle management:

- **Structured logging** is configured first, before anything else logs.
- **Lifespan** performs pre-flight warm-up so the first patient request
  never pays a cold start: pgvector verification, Groq/LLM warm-up,
  embedding-model load, and DB-pool priming - each best-effort, none
  fatal to startup.
- **Middleware** stack (outermost first): request correlation/logging ->
  timing/NFR-02 -> CORS.
- **/metrics** exposes the Prometheus registry for scraping.
- **Global exception handlers** guarantee every error path still returns
  the clinical safety disclaimer and never leaks internals.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.ai.readiness import health_payload
from app.api.v1.endpoints import ai, auth
from app.core.config import settings
from app.core.logging import (
    RequestLoggingMiddleware,
    configure_structured_logging,
    get_correlation_id,
)
from app.core.middleware import TimingMiddleware
from app.core.monitoring import (
    render_prometheus,
    set_build_info,
)
from app.core.security import validate_security_settings
from app.utils.constants import SAFETY_DISCLAIMER

# Refuse to build the app with a missing/short SECRET_KEY (outside tests).
validate_security_settings(settings)

# Configure logging before any module-level logger emits a line.
configure_structured_logging()
logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Lifespan: startup warm-up + graceful shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    On startup, run four best-effort warm-up steps on the serving chatbot
    (pgvector check, LLM clients, DB pool, embedding model + one FAQ query)
    so the first request is served hot. uvicorn runs this before it opens
    the port. Each step is isolated: a failure is logged and startup
    proceeds; the RAG step records ``rag_ready`` for the health endpoints.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control returns to the server for the serving phase.
    """
    logger.info(
        "Hoku Health Care API starting (version=%s, env=%s)",
        APP_VERSION,
        settings.ENVIRONMENT,
    )
    set_build_info(version=APP_VERSION, environment=settings.ENVIRONMENT)

    await _verify_pgvector()
    await _warm_up_chatbot()
    await _prime_db_pool()
    await _warm_up_rag()

    logger.info("Startup warm-up complete; ready to accept traffic.")
    yield

    logger.info("Hoku Health Care API shutting down.")


async def _verify_pgvector() -> None:
    """Log whether the native pgvector path or the SQLite fallback is active."""
    try:
        from sqlalchemy import text

        from app.core.database import engine

        if engine.dialect.name != "postgresql":
            logger.info(
                "pgvector check skipped: dialect=%s (cosine fallback)",
                engine.dialect.name,
            )
            return

        def _check() -> bool:
            with engine.connect() as conn:
                return bool(
                    conn.execute(
                        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
                    ).scalar()
                )

        installed = await asyncio.to_thread(_check)
        if installed:
            logger.info("pgvector extension verified.")
        else:
            logger.warning(
                "pgvector extension NOT installed - RAG grounding will be inactive. "
                "Run: CREATE EXTENSION IF NOT EXISTS vector;"
            )
    except Exception as exc:  # noqa: BLE001 - warm-up must never abort startup
        logger.warning(
            "pgvector verification failed (non-fatal): %s", type(exc).__name__
        )


async def _warm_up_chatbot() -> None:
    """Initialize the serving chatbot's Groq clients and LLM chain."""
    try:
        from app.services.ai_service import _chatbot

        await asyncio.to_thread(_chatbot.warm_up)
        logger.info("Chatbot/LLM warm-up completed.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Chatbot warm-up failed (will retry lazily): %s", type(exc).__name__
        )


async def _warm_up_rag() -> None:
    """
    Load the embedding model and query the FAQ index through the serving
    chatbot's RAG, so the first FAQ question is warm; records ``rag_ready``.
    """
    from app.ai.readiness import check_rag, set_rag_ready

    try:
        from app.services.ai_service import _chatbot

        ready = await asyncio.to_thread(check_rag, _chatbot.rag)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG warm-up failed: %s", type(exc).__name__)
        set_rag_ready(False)
        return
    if ready:
        logger.info("RAG warm-up completed: embedding model and FAQ index ready.")
    else:
        logger.warning(
            "RAG warm-up found no FAQ content (model, FAQ rows or vector "
            "query unavailable); FAQ questions will take the no-match path."
        )


async def _prime_db_pool() -> None:
    """Open and return one pooled connection to remove first-hit latency."""
    try:
        from sqlalchemy import text

        from app.core.database import engine

        def _ping() -> None:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))

        await asyncio.to_thread(_ping)
        logger.info("Database pool primed.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB pool priming failed (non-fatal): %s", type(exc).__name__)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Hoku Health Care API",
    description=(
        "AI-powered home healthcare assistance platform.\n\n"
        "The chatbot never provides a definitive diagnosis; every response "
        'ends with *"Please consult a doctor for proper diagnosis."*'
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "AI Chatbot", "description": "Patient-facing chat and history."},
        {"name": "Doctors", "description": "Specialist lookup and availability."},
        {
            "name": "RAG (admin)",
            "description": "FAQ vector-store seeding and debug search.",
        },
        {"name": "Monitoring", "description": "Safety and performance metrics."},
        {"name": "Health", "description": "Liveness probes."},
    ],
)

# ------------------------------------------------------------------
# Middleware (added last = outermost). Order of execution per request:
#   RequestLogging -> Timing -> CORS -> route
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=[
        "X-Hoku-Emergency",
        "X-Hoku-Emergency-Severity",
        "X-Response-Time-Sec",
        "X-Correlation-ID",
        "RateLimit-Limit",
        "RateLimit-Remaining",
        "RateLimit-Reset",
        "Retry-After",
    ],
)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestLoggingMiddleware)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """
    Render HTTP exceptions as JSON with the correlation ID attached.

    Args:
        request: The request that raised.
        exc: The HTTP exception.

    Returns:
        JSONResponse: ``{"detail": ..., "correlation_id": ...}``.
    """
    headers = getattr(exc, "headers", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "correlation_id": get_correlation_id()},
        headers=headers,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Return 422 with structured validation errors.

    Args:
        request: The request that failed validation.
        exc: The validation error.

    Returns:
        JSONResponse: The error list plus correlation ID. On /api/auth/*
        the submitted values ("input"/"ctx") are dropped so a password is
        never echoed back.
    """
    errors = exc.errors()
    if request.url.path.startswith("/api/auth"):
        errors = [
            {k: v for k, v in e.items() if k not in ("input", "ctx")} for e in errors
        ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": errors,
            "correlation_id": get_correlation_id(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all for unexpected errors.

    Logs the full traceback server-side but returns a generic,
    disclaimer-bearing message so patients never see a raw stack trace
    and PHI never leaks into the response body.

    Args:
        request: The request that raised.
        exc: The unhandled exception.

    Returns:
        JSONResponse: A safe 500 payload.
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": (
                "An unexpected error occurred. Please try again shortly. "
                f"{SAFETY_DISCLAIMER}"
            ),
            "correlation_id": get_correlation_id(),
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------
app.include_router(auth.router)
app.include_router(ai.router)


# ---------------------------------------------------------------------------
# Root & metrics
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"], summary="Root health check")
async def root() -> dict:
    """
    Root liveness endpoint.

    Returns:
        dict: Service name, version, and environment.
    """
    return {
        "message": "Hoku Health Care API is running",
        "version": APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Readiness probe (same payload as /api/ai/health)",
)
async def health() -> dict:
    """``{"status": "ok" | "degraded", "service", "rag_ready"}``, always 200."""
    return health_payload()


@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Prometheus metrics",
    include_in_schema=False,
)
async def metrics(request: Request) -> Response:
    """
    Expose the Prometheus text exposition format for scraping.

    - ``METRICS_ENABLED=false``: 404.
    - ``METRICS_TOKEN`` set: requires ``Authorization: Bearer <METRICS_TOKEN>``
      (constant-time compare); missing or wrong -> 401.
    - ``METRICS_TOKEN`` unset: open in ENVIRONMENT test/development only,
      404 everywhere else.

    Returns:
        Response: The metrics payload, or 401/404 as above.
    """
    if not settings.METRICS_ENABLED:
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    if settings.METRICS_TOKEN:
        supplied = request.headers.get("authorization", "")
        expected = f"Bearer {settings.METRICS_TOKEN}"
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            return Response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={"WWW-Authenticate": "Bearer"},
            )
    elif settings.ENVIRONMENT.lower() not in ("test", "development"):
        return Response(status_code=status.HTTP_404_NOT_FOUND)
    payload, content_type = render_prometheus()
    return Response(content=payload, media_type=content_type)
