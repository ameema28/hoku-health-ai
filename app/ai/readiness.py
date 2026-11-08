"""
Startup readiness of FAQ retrieval (reported by the health endpoints).

``rag_ready`` is True only after a startup warm-up query found at least one
FAQ through the serving chatbot's RAG pipeline. That needs the embedding
model loaded (a zero-vector fallback finds nothing), FAQ rows in the
database and, on PostgreSQL, a working pgvector column. When it is False the
app still answers; FAQ questions then take the no-match path.
"""

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

#: A paraphrase of a seeded FAQ, so a ready index always returns a row.
WARM_UP_QUERY = "How do I book a home visit?"

_lock = threading.Lock()
_rag_ready = False


def rag_ready() -> bool:
    """Whether the last warm-up found FAQ content."""
    with _lock:
        return _rag_ready


def set_rag_ready(value: bool) -> None:
    """Record the warm-up result (also used by tests)."""
    global _rag_ready
    with _lock:
        _rag_ready = bool(value)


def check_rag(rag: Any) -> bool:
    """
    Run one similarity search through ``rag`` and record the result.

    ``similarity_search`` returns no rows when the model is unavailable,
    the table is empty or the vector query fails, so one call covers all
    three. Never raises.
    """
    try:
        ready = bool(rag.similarity_search(WARM_UP_QUERY, k=1))
    except Exception as exc:  # noqa: BLE001 - readiness must never raise
        logger.warning("RAG readiness check failed: %s", type(exc).__name__)
        ready = False
    set_rag_ready(ready)
    return ready


def health_payload() -> dict:
    """Body of GET /health and GET /api/ai/health."""
    ready = rag_ready()
    return {
        "status": "ok" if ready else "degraded",
        "service": "Hoku AI Chatbot",
        "rag_ready": ready,
    }
