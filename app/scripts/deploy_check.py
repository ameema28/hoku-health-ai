"""
Hoku Health Care - Pre-flight Deployment Verification (Day 10).

Run this immediately after any deploy, before routing patient traffic::

    python -m app.scripts.deploy_check
    python -m app.scripts.deploy_check --strict     # warnings become failures
    python -m app.scripts.deploy_check --live-groq  # spend one real Groq call

Checks performed, in dependency order:

1. **Environment variables** - every setting the runtime actually reads,
   sourced from ``app.core.config.settings`` and ``app.ai.config.ai_settings``
   rather than ``os.environ``, so ``.env`` precedence is respected.
2. **Database connectivity** - ``SELECT 1`` through the existing
   ``app.core.database.engine`` (validates the pool, not just the URL).
3. **Groq reachability** - key format by default; a real one-token call
   with ``--live-groq``.
4. **pgvector** - extension presence and the ``vector`` type. Reported as a
   WARNING on SQLite, where ``app/ai/rag.py`` legitimately falls back to an
   in-Python cosine scan.
5. **Vector store** - the ``vector_store`` table exists and is populated;
   an empty store means RAG silently answers from general knowledge.
6. **Clinical safety invariants** - the mandatory disclaimer constant is
   intact and EMERGENCY/SYMPTOM intents are still excluded from caching.

Exit codes:
    0 - all checks passed (warnings may be present)
    1 - at least one check FAILED
    2 - the checker itself crashed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deploy_check")


class Status(str, Enum):
    """Outcome of a single deployment check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class CheckResult:
    """
    Result of one verification step.

    Attributes:
        name: Human-readable check name.
        status: PASS, WARN, or FAIL.
        detail: Short explanation shown in the report.
        elapsed_ms: Wall-clock duration of the check.
    """

    name: str
    status: Status
    detail: str
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def check_environment() -> CheckResult:
    """
    Verify that every required setting is present and non-default.

    Reads through the Pydantic settings objects the application itself
    uses, so a value supplied by ``.env``, Docker, or Render is seen
    exactly as the runtime will see it.

    Returns:
        CheckResult: FAIL when a required secret is missing or still
        carries its insecure development default.
    """
    from app.ai.config import ai_settings
    from app.core.config import settings

    missing: List[str] = []
    insecure: List[str] = []

    if not ai_settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    elif not ai_settings.groq_api_key.startswith("gsk_"):
        insecure.append("GROQ_API_KEY (expected a 'gsk_' prefix)")

    if not settings.DATABASE_URL:
        missing.append("DATABASE_URL")

    if settings.is_production:
        if settings.SECRET_KEY == "change-me-in-production":
            insecure.append("SECRET_KEY is still the development default")
        if settings.DEBUG:
            insecure.append("DEBUG=true in a production environment")
        if settings.DATABASE_URL.startswith("sqlite"):
            insecure.append("DATABASE_URL points at SQLite in production")

    if missing:
        return CheckResult(
            "Environment variables",
            Status.FAIL,
            f"Missing: {', '.join(missing)}",
        )
    if insecure:
        return CheckResult(
            "Environment variables",
            Status.FAIL if settings.is_production else Status.WARN,
            "; ".join(insecure),
        )
    return CheckResult(
        "Environment variables",
        Status.PASS,
        f"env={settings.ENVIRONMENT}, models={ai_settings.GROQ_FAST_MODEL}/"
        f"{ai_settings.GROQ_MAIN_MODEL}",
    )


def check_database() -> CheckResult:
    """
    Confirm the database is reachable through the application engine.

    Uses ``app.core.database.engine`` so the configured QueuePool,
    ``pool_pre_ping`` and ``connect_args`` are all exercised.

    Returns:
        CheckResult: PASS with the server version, FAIL on any
        connection or authentication error.
    """
    from sqlalchemy import text

    from app.core.database import engine

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            dialect = engine.dialect.name
            if dialect == "postgresql":
                version = conn.execute(text("SHOW server_version")).scalar()
                detail = f"PostgreSQL {version} via {engine.pool.__class__.__name__}"
            else:
                detail = f"{dialect} via {engine.pool.__class__.__name__}"
        return CheckResult("Database connectivity", Status.PASS, detail)
    except Exception as exc:  # noqa: BLE001 - report any failure verbatim
        return CheckResult(
            "Database connectivity",
            Status.FAIL,
            f"{type(exc).__name__}: {exc}",
        )


def check_groq(live: bool = False) -> CheckResult:
    """
    Verify the Groq client can be constructed, and optionally that the
    key is accepted by the API.

    Args:
        live: When True, issue a single 1-token completion against the
            fast model. Costs a real API call; off by default so the
            checker is safe to run in CI.

    Returns:
        CheckResult: PASS when the client initializes (and responds, if
        ``live``), FAIL when LangChain/Groq is unavailable or the key is
        rejected.
    """
    from app.ai.config import ai_settings

    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        return CheckResult(
            "Groq client",
            Status.FAIL,
            f"langchain-groq not installed: {exc}",
        )

    if not ai_settings.groq_api_key:
        return CheckResult("Groq client", Status.FAIL, "GROQ_API_KEY is empty")

    try:
        llm = ChatGroq(
            groq_api_key=ai_settings.groq_api_key,
            model_name=ai_settings.GROQ_FAST_MODEL,
            temperature=0.0,
            max_tokens=1,
            request_timeout=ai_settings.GROQ_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Groq client",
            Status.FAIL,
            f"Client init failed: {type(exc).__name__}: {exc}",
        )

    if not live:
        return CheckResult(
            "Groq client",
            Status.PASS,
            f"Client initialized for {ai_settings.GROQ_FAST_MODEL} "
            "(pass --live-groq to verify the key against the API)",
        )

    try:
        started = time.perf_counter()
        llm.invoke("ping")
        elapsed = (time.perf_counter() - started) * 1000
        return CheckResult(
            "Groq client",
            Status.PASS,
            f"Live call succeeded in {elapsed:.0f}ms",
            elapsed_ms=elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "Groq client",
            Status.FAIL,
            f"Live call rejected: {type(exc).__name__}: {exc}",
        )


def check_pgvector() -> CheckResult:
    """
    Verify the pgvector extension is installed on PostgreSQL.

    On SQLite this is reported as a WARNING, not a failure: Day 5's RAG
    layer deliberately falls back to an in-Python cosine-similarity scan
    when pgvector is unavailable.

    Returns:
        CheckResult: PASS on PostgreSQL with the extension enabled, WARN
        on SQLite, FAIL on PostgreSQL without the extension.
    """
    from sqlalchemy import text

    from app.core.database import engine

    if engine.dialect.name != "postgresql":
        return CheckResult(
            "pgvector extension",
            Status.WARN,
            f"Dialect is '{engine.dialect.name}' - RAG will use the "
            "in-Python cosine fallback (expected in local dev)",
        )

    try:
        with engine.connect() as conn:
            installed = conn.execute(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            ).scalar()
            if not installed:
                return CheckResult(
                    "pgvector extension",
                    Status.FAIL,
                    "Extension 'vector' is not installed. Run: "
                    "CREATE EXTENSION IF NOT EXISTS vector;",
                )
        return CheckResult(
            "pgvector extension",
            Status.PASS,
            f"vector {installed} enabled",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "pgvector extension",
            Status.FAIL,
            f"{type(exc).__name__}: {exc}",
        )


def check_vector_store() -> CheckResult:
    """
    Verify the FAQ vector store table exists and holds documents.

    An empty store is a WARNING rather than a failure: the chatbot still
    answers from general LLM knowledge, but the entire Day 5 deliverable
    is inert until ``python -m app.scripts.seed_faqs`` has been run.

    Returns:
        CheckResult: PASS with the row count, WARN if empty or missing.
    """
    from sqlalchemy import inspect, text

    from app.core.config import settings
    from app.core.database import engine

    try:
        inspector = inspect(engine)
        if "vector_store" not in inspector.get_table_names():
            return CheckResult(
                "FAQ vector store",
                Status.WARN,
                "Table 'vector_store' not found - run init_db.py, then "
                "python -m app.scripts.seed_faqs",
            )

        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM vector_store")).scalar() or 0

        if count == 0:
            return CheckResult(
                "FAQ vector store",
                Status.WARN,
                "Vector store is empty - RAG grounding is inactive. "
                "Run: python -m app.scripts.seed_faqs",
            )

        return CheckResult(
            "FAQ vector store",
            Status.PASS,
            f"{count} documents in collection '{settings.COLLECTION_NAME}' "
            f"(dim={settings.VECTOR_DIMENSION}, "
            f"threshold={settings.RAG_SIMILARITY_THRESHOLD})",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult(
            "FAQ vector store",
            Status.FAIL,
            f"{type(exc).__name__}: {exc}",
        )


def check_clinical_safety() -> CheckResult:
    """
    Assert the two non-negotiable clinical invariants still hold.

    1. ``SAFETY_DISCLAIMER`` is exactly the sentence every response path
       must end with.
    2. ``ResponseCache.should_cache`` still refuses EMERGENCY and
       SYMPTOM intents - a regression here would let a stale answer be
       served for a live clinical query.

    Returns:
        CheckResult: FAIL if either invariant has drifted.
    """
    from app.ai.caching import ResponseCache
    from app.utils.constants import SAFETY_DISCLAIMER

    expected = "Please consult a doctor for proper diagnosis."
    problems: List[str] = []

    if SAFETY_DISCLAIMER != expected:
        problems.append(f"SAFETY_DISCLAIMER drifted: {SAFETY_DISCLAIMER!r}")

    cache = ResponseCache()
    if cache.should_cache("emergency", is_emergency=True):
        problems.append("EMERGENCY intent is cacheable")
    if cache.should_cache("emergency", is_emergency=False):
        problems.append("EMERGENCY intent is cacheable without the emergency flag")
    if cache.should_cache("symptom", is_emergency=False):
        problems.append("SYMPTOM intent is cacheable")

    if problems:
        return CheckResult("Clinical safety invariants", Status.FAIL, "; ".join(problems))

    return CheckResult(
        "Clinical safety invariants",
        Status.PASS,
        "Disclaimer intact; EMERGENCY and SYMPTOM excluded from cache",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_checks(live_groq: bool = False) -> List[CheckResult]:
    """
    Execute every deployment check in dependency order.

    Args:
        live_groq: Forwarded to :func:`check_groq`.

    Returns:
        List[CheckResult]: One result per check, in execution order.
    """
    checks: List[Tuple[str, Callable[[], CheckResult]]] = [
        ("environment", check_environment),
        ("database", check_database),
        ("groq", lambda: check_groq(live=live_groq)),
        ("pgvector", check_pgvector),
        ("vector_store", check_vector_store),
        ("clinical_safety", check_clinical_safety),
    ]

    results: List[CheckResult] = []
    for key, fn in checks:
        started = time.perf_counter()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - never let one check abort the run
            result = CheckResult(
                key,
                Status.FAIL,
                f"Checker crashed: {type(exc).__name__}: {exc}",
            )
        if not result.elapsed_ms:
            result.elapsed_ms = (time.perf_counter() - started) * 1000
        results.append(result)
    return results


def render_report(results: List[CheckResult]) -> str:
    """
    Format results as an aligned, human-readable console table.

    Args:
        results: Output of :func:`run_all_checks`.

    Returns:
        str: The full multi-line report.
    """
    icons = {Status.PASS: "[PASS]", Status.WARN: "[WARN]", Status.FAIL: "[FAIL]"}
    width = max(len(r.name) for r in results) + 2

    lines = [
        "",
        "=" * 78,
        " HOKU HEALTH CARE - DEPLOYMENT PRE-FLIGHT CHECK",
        "=" * 78,
    ]
    for r in results:
        lines.append(f"{icons[r.status]} {r.name.ljust(width)} {r.detail}")
        lines.append(f"{'':7}{'':{width}} ({r.elapsed_ms:.0f}ms)")

    passed = sum(1 for r in results if r.status is Status.PASS)
    warned = sum(1 for r in results if r.status is Status.WARN)
    failed = sum(1 for r in results if r.status is Status.FAIL)

    lines += [
        "-" * 78,
        f" {passed} passed | {warned} warnings | {failed} failed",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    """
    CLI entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        int: Process exit code - 0 on success, 1 on failure, 2 on crash.
    """
    parser = argparse.ArgumentParser(
        prog="deploy_check",
        description="Verify a Hoku Health Care deployment before routing traffic.",
    )
    parser.add_argument(
        "--live-groq",
        action="store_true",
        help="Issue one real Groq completion to validate the API key.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (recommended for production gates).",
    )
    args = parser.parse_args(argv)

    try:
        results = run_all_checks(live_groq=args.live_groq)
    except Exception as exc:  # noqa: BLE001
        logger.exception("deploy_check crashed: %s", exc)
        return 2

    print(render_report(results))

    if any(r.status is Status.FAIL for r in results):
        logger.error("Deployment check FAILED - do not route patient traffic.")
        return 1
    if args.strict and any(r.status is Status.WARN for r in results):
        logger.error("Deployment check failed under --strict (warnings present).")
        return 1

    logger.info("Deployment check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())