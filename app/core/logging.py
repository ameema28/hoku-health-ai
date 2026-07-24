"""
Hoku Health Care - Structured Logging & Request Correlation (Day 10).

Replaces the ad-hoc ``logging.basicConfig`` format from
``app/core/config.py::configure_logging`` with machine-parseable JSON,
so Render's log drain, Grafana Loki, or CloudWatch Insights can query
fields instead of regexing free text.

Three pieces:

1. :class:`JSONFormatter` - one JSON object per line, with the
   correlation ID and any ``extra=`` fields folded in.
2. :class:`RedactionFilter` - a hard barrier against PHI in logs. Patient
   messages, AI replies, emails, phone numbers, JWTs and API keys are
   scrubbed *before* the record reaches any handler.
3. :class:`RequestLoggingMiddleware` - assigns a correlation ID per
   request, times it, echoes ``X-Correlation-ID`` back to the caller,
   and emits one structured access log line.

Clinical-privacy rule
---------------------
The chatbot handles symptom descriptions, which are Protected Health
Information. **Never log ``user_message``, ``raw_message``, ``reply``,
``ai_response`` or FAQ context.** ``RedactionFilter`` enforces this even
when a future call site forgets: those keys are dropped from ``extra``
outright, and free-text patterns are masked. ``user_id`` is retained
because it is a pseudonymous internal key needed for support triage.

Usage::

    from app.core.logging import configure_structured_logging
    configure_structured_logging()   # once, at import time in main.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, Final, Iterable, Optional, Pattern, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Correlation ID propagation
# ---------------------------------------------------------------------------
#: Per-request identifier, readable from any coroutine in the same task
#: tree without threading it through every function signature.
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="-")

#: Authenticated user id for the current request, when known.
user_id_ctx: ContextVar[Optional[int]] = ContextVar("user_id", default=None)

CORRELATION_HEADER: Final[str] = "X-Correlation-ID"


def get_correlation_id() -> str:
    """
    Return the correlation ID bound to the current request context.

    Returns:
        str: The current ID, or ``"-"`` outside a request.
    """
    return correlation_id_ctx.get()


def set_correlation_id(value: Optional[str] = None) -> str:
    """
    Bind a correlation ID to the current context.

    Args:
        value: An inbound ID to reuse (e.g. from an upstream gateway).
            When None or blank, a fresh UUID4 hex is generated.

    Returns:
        str: The ID that was bound.
    """
    resolved = value.strip() if value and value.strip() else uuid.uuid4().hex
    correlation_id_ctx.set(resolved)
    return resolved


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
#: Structured-logging keys that must never be persisted. Anything landing
#: in ``extra={...}`` under one of these names is dropped entirely.
FORBIDDEN_EXTRA_KEYS: Final[frozenset] = frozenset(
    {
        "message_text",
        "user_message",
        "raw_message",
        "clean_message",
        "reply",
        "ai_response",
        "faq_context",
        "symptoms",
        "chat_history",
        "password",
        "token",
        "access_token",
        "authorization",
        "api_key",
        "groq_api_key",
        "secret_key",
        "email",
        "phone",
        "address",
    }
)

_REDACTED: Final[str] = "[REDACTED]"

#: Free-text patterns scrubbed from the rendered log message itself.
_REDACTION_PATTERNS: Final[Tuple[Tuple[Pattern[str], str], ...]] = (
    # Groq / OpenAI style API keys
    (re.compile(r"\bgsk_[A-Za-z0-9]{10,}\b"), _REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), _REDACTED),
    # JWTs (three base64url segments)
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\b"), _REDACTED),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer " + _REDACTED),
    # Email addresses
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), _REDACTED),
    # Phone numbers, incl. PK/UAE/UK international forms
    (re.compile(r"(?<!\d)\+?\d[\d\s\-().]{8,}\d(?!\d)"), _REDACTED),
    # Postgres URLs with inline credentials
    (re.compile(r"(?i)(postgres(?:ql)?://)[^:@\s]+:[^@\s]+@"), r"\1" + _REDACTED + "@"),
)


class RedactionFilter(logging.Filter):
    """
    Logging filter that strips PHI and secrets from every record.

    Applied at the *filter* stage, so it runs before any handler
    formats or ships the record. Two passes:

    1. ``extra`` attributes whose key is in :data:`FORBIDDEN_EXTRA_KEYS`
       are replaced with ``[REDACTED]``.
    2. The interpolated message text is run through
       :data:`_REDACTION_PATTERNS`.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Redact the record in place.

        Args:
            record: The log record about to be emitted.

        Returns:
            bool: Always True - records are scrubbed, never dropped, so
            that operational signal is preserved.
        """
        for key in list(vars(record).keys()):
            if key.lower() in FORBIDDEN_EXTRA_KEYS:
                setattr(record, key, _REDACTED)

        try:
            rendered = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken format string must not kill logging
            return True

        redacted = rendered
        for pattern, replacement in _REDACTION_PATTERNS:
            redacted = pattern.sub(replacement, redacted)

        if redacted != rendered:
            record.msg = redacted
            record.args = ()

        return True


def redact(value: str) -> str:
    """
    Apply the redaction patterns to an arbitrary string.

    Useful when a value must be echoed into an error response or a
    non-logging sink.

    Args:
        value: Raw text that may contain PHI or secrets.

    Returns:
        str: The redacted text.
    """
    result = value
    for pattern, replacement in _REDACTION_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------
#: Standard LogRecord attributes, excluded when harvesting ``extra``.
_RESERVED_ATTRS: Final[frozenset] = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
    }
)


class JSONFormatter(logging.Formatter):
    """
    Render log records as single-line JSON objects.

    Emitted shape::

        {
          "timestamp": "2026-07-24T18:04:11.912Z",
          "level": "INFO",
          "logger": "app.ai.chatbot",
          "message": "Intent=general, confidence=0.88",
          "correlation_id": "9f2c...",
          "user_id": 123,
          "module": "chatbot",
          "line": 512,
          "service": "hoku-health-backend",
          "environment": "production"
        }

    Attributes:
        service_name: Value written to the ``service`` field.
        environment: Value written to the ``environment`` field.
    """

    def __init__(
        self,
        service_name: str = "hoku-health-backend",
        environment: str = "development",
    ) -> None:
        """
        Initialize the formatter.

        Args:
            service_name: Logical service identifier.
            environment: Deployment environment name.
        """
        super().__init__()
        self.service_name = service_name
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        """
        Serialize a record to a JSON string.

        Args:
            record: The record to format.

        Returns:
            str: One-line JSON payload.
        """
        payload: Dict[str, Any] = {
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            )
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": get_correlation_id(),
            "module": record.module,
            "line": record.lineno,
            "service": self.service_name,
            "environment": self.environment,
        }

        current_user = user_id_ctx.get()
        if current_user is not None:
            payload["user_id"] = current_user

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Fold in any extra={...} fields that survived redaction.
        for key, value in vars(record).items():
            if key in _RESERVED_ATTRS or key.startswith("_"):
                continue
            if key in payload:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)

        return json.dumps(payload, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Configuration entry point
# ---------------------------------------------------------------------------
def configure_structured_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    service_name: str = "hoku-health-backend",
    quiet_loggers: Iterable[str] = ("uvicorn.access", "httpx", "httpcore", "urllib3"),
) -> None:
    """
    Install the JSON formatter and redaction filter on the root logger.

    Idempotent: existing handlers are removed first, so calling this
    after uvicorn has configured its own handlers yields a single,
    consistent stream.

    Args:
        level: Log level name. Defaults to ``$LOG_LEVEL`` or ``INFO``.
        log_format: ``"json"`` (default) or ``"plain"``. ``$LOG_FORMAT``
            is consulted when omitted; ``plain`` keeps the Day 0-9
            human-readable format for local debugging.
        service_name: Value for the ``service`` field.
        quiet_loggers: Chatty third-party loggers pinned to WARNING.
    """
    from app.core.config import settings

    resolved_level = (level or os.getenv("LOG_LEVEL") or "").upper()
    if not resolved_level:
        resolved_level = "DEBUG" if settings.DEBUG else "INFO"

    resolved_format = (log_format or os.getenv("LOG_FORMAT") or "json").lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RedactionFilter())

    if resolved_format == "json":
        handler.setFormatter(
            JSONFormatter(service_name=service_name, environment=settings.ENVIRONMENT)
        )
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved_level, logging.INFO))

    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Structured logging configured (format=%s, level=%s, env=%s)",
        resolved_format,
        resolved_level,
        settings.ENVIRONMENT,
    )


# ---------------------------------------------------------------------------
# Request middleware
# ---------------------------------------------------------------------------
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Per-request correlation, timing, and structured access logging.

    Complements ``app.core.middleware.TimingMiddleware`` (which owns the
    ``X-Response-Time-Sec`` header and the NFR-02 breach alerts) rather
    than replacing it: this one owns identity and the access log line.

    Behaviour:
        * Reuses an inbound ``X-Correlation-ID`` when present, otherwise
          mints a UUID4 - so a trace can be followed from the React app
          through to the Groq call.
        * Echoes the ID back on every response, including error paths.
        * Logs method, path, status, and duration. Query strings and
          bodies are never logged: symptom text frequently arrives in
          both.
    """

    #: Paths excluded from access logging to keep scrape noise down.
    SKIP_PATHS: frozenset = frozenset({"/metrics", "/api/ai/health", "/favicon.ico"})

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Bind a correlation ID, time the request, and log the outcome.

        Args:
            request: Incoming request.
            call_next: Downstream ASGI handler.

        Returns:
            Response: The downstream response with ``X-Correlation-ID`` set.
        """
        cid = set_correlation_id(request.headers.get(CORRELATION_HEADER))
        user_id_ctx.set(None)

        path = request.url.path
        started = time.perf_counter()
        logger = logging.getLogger("app.access")

        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "Unhandled exception: %s %s after %.1fms",
                request.method,
                path,
                elapsed_ms,
                extra={"http_method": request.method, "http_path": path},
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers[CORRELATION_HEADER] = cid

        if path not in self.SKIP_PATHS:
            log_at = logging.WARNING if response.status_code >= 500 else logging.INFO
            logger.log(
                log_at,
                "%s %s -> %d in %.1fms",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
                extra={
                    "http_method": request.method,
                    "http_path": path,
                    "http_status": response.status_code,
                    "duration_ms": round(elapsed_ms, 2),
                },
            )

        return response