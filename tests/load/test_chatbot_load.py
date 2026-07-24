"""
Hoku Health Care - Chatbot Load Tests (Day 10).

Two entry points in one file:

1. **pytest** (default, CI-safe, Groq mocked): ``test_50_concurrent_users``
   fires 50 concurrent chat requests through an in-process ASGI transport
   with the LLM boundary patched, and asserts:
       * zero HTTP 500s,
       * zero rate-limit-induced failures (limiter disabled for the run),
       * P95 latency < 4s (NFR-02).
   This is the version that runs in CI. It never touches the network.

2. **Locust** (manual, against a real deployment): ``HokuChatUser`` drives
   sustained traffic at a live URL for staging/soak testing::

       locust -f tests/load/test_chatbot_load.py \\
              --host https://hoku-health-backend.onrender.com \\
              --users 50 --spawn-rate 10 --run-time 3m

   Against a real host the built-in 5 req/min limit applies, so Locust is
   for realistic rate-limited soak behaviour, not raw throughput.

The pytest path is intentionally the CI gate; the Locust class is skipped
when Locust is not installed.
"""

from __future__ import annotations

import datetime
import json
import os
import statistics
import time
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from jose import jwt

CONCURRENCY = 50
NFR02_CEILING_SECONDS = 4.0

# ---------------------------------------------------------------------------
# Shared token helper (mirrors token_gen.py)
# ---------------------------------------------------------------------------
def _make_token(user_id: int) -> str:
    """
    Mint a Bearer token for a synthetic load-test user.

    Args:
        user_id: The subject id.

    Returns:
        str: ``Bearer <jwt>``.
    """
    from app.core.config import settings

    token = jwt.encode(
        {
            "sub": str(user_id),
            "id": user_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2),
        },
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return f"Bearer {token}"


def _mock_chain_return() -> Dict[str, str]:
    """Return a LangChain-shaped ``{"text": <json>}`` payload."""
    from app.utils.constants import SAFETY_DISCLAIMER

    return {
        "text": json.dumps(
            {
                "reply": "Here is some general wellness guidance. " + SAFETY_DISCLAIMER,
                "suggestedSpecialist": "General Physician",
                "severity": "mild",
                "shouldSeeDoctor": False,
            }
        )
    }


# ===========================================================================
# pytest path (Groq mocked, CI gate)
# ===========================================================================
@pytest.mark.asyncio
async def test_50_concurrent_users() -> None:
    """
    Drive 50 concurrent chat requests and assert NFR-02 P95 + zero 500s.

    Rate limiting is disabled for the run via a patched setting, because
    the point here is pipeline behaviour under concurrency, not the
    5 req/min product limit (which is covered separately in the
    integration suite). Groq is fully mocked.
    """
    import asyncio

    import httpx
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.core.database import Base, get_db

    # ---- Isolated SQLite DB for the load run ----------------------------
    db_url = os.getenv("LOAD_TEST_DATABASE_URL", "sqlite:///./test_load.db")
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def _override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    chain = MagicMock()
    chain.invoke = MagicMock(return_value=_mock_chain_return())

    # Import the app lazily so the setting patch below is applied first.
    with patch("app.core.config.settings.RATE_LIMIT_ENABLED", False), patch(
        "app.ai.chatbot.LLMChain", return_value=chain
    ):
        from app.main import app

        app.dependency_overrides[get_db] = _override_get_db

        latencies: List[float] = []
        statuses: List[int] = []

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://loadtest"
        ) as async_client:

            async def _one_request(idx: int) -> Tuple[int, float]:
                headers = {"Authorization": _make_token(user_id=idx + 1)}
                started = time.perf_counter()
                resp = await async_client.post(
                    "/api/ai/chat",
                    json={"message": "Give me a wellness tip", "userId": idx + 1},
                    headers=headers,
                )
                return resp.status_code, time.perf_counter() - started

            results = await asyncio.gather(
                *(_one_request(i) for i in range(CONCURRENCY))
            )

        app.dependency_overrides.clear()

    for status_code, elapsed in results:
        statuses.append(status_code)
        latencies.append(elapsed)

    # ---- Assertions -----------------------------------------------------
    server_errors = [s for s in statuses if s >= 500]
    assert not server_errors, f"{len(server_errors)} requests returned 5xx"

    ok = [s for s in statuses if s == 200]
    assert len(ok) == CONCURRENCY, f"only {len(ok)}/{CONCURRENCY} returned 200: {statuses}"

    latencies.sort()
    p95_index = min(int(len(latencies) * 0.95), len(latencies) - 1)
    p95 = latencies[p95_index]
    p50 = statistics.median(latencies)

    assert p95 < NFR02_CEILING_SECONDS, (
        f"P95 latency {p95:.3f}s exceeded the {NFR02_CEILING_SECONDS}s NFR-02 ceiling "
        f"(P50={p50:.3f}s, max={latencies[-1]:.3f}s)"
    )


# ===========================================================================
# Locust path (manual, real host)
# ===========================================================================
try:
    from locust import HttpUser, between, task

    LOCUST_AVAILABLE = True
except ImportError:  # pragma: no cover - Locust optional in CI
    LOCUST_AVAILABLE = False
    HttpUser = object  # type: ignore[assignment,misc]

    def task(*args: Any, **kwargs: Any):  # type: ignore[misc]
        """No-op decorator stub when Locust is absent."""
        def _decorator(fn):
            return fn

        return _decorator

    def between(a: float, b: float):  # type: ignore[misc]
        """No-op stub when Locust is absent."""
        return None


if LOCUST_AVAILABLE:

    class HokuChatUser(HttpUser):
        """
        A simulated patient hitting the live chat endpoint.

        Run against a deployed host only. The real 5 req/min per-user
        limit means each simulated user will see 429s under sustained
        load - that is expected and correct behaviour, so the on_start
        token is unique per spawned user to exercise many buckets.
        """

        wait_time = between(1, 3)

        _MESSAGES: List[str] = [
            "What services does Hoku offer?",
            "I have a mild headache, any advice?",
            "How do I book an appointment?",
            "Do you provide home nursing?",
            "What are your visiting hours?",
        ]

        def on_start(self) -> None:
            """Assign a unique user id + token when this virtual user spawns."""
            # Spread ids widely so limiter buckets don't collide across users.
            self._user_id = int(time.time() * 1000) % 1_000_000
            self._headers = {"Authorization": _make_token(self._user_id)}
            self._i = 0

        @task
        def send_chat(self) -> None:
            """Send one chat request and record non-2xx/429 as failures."""
            message = self._MESSAGES[self._i % len(self._MESSAGES)]
            self._i += 1
            with self.client.post(
                "/api/ai/chat",
                json={"message": message, "userId": self._user_id},
                headers=self._headers,
                catch_response=True,
                name="/api/ai/chat",
            ) as response:
                if response.status_code == 429:
                    # Expected under the product rate limit; not a failure.
                    response.success()
                elif response.status_code >= 500:
                    response.failure(f"5xx: {response.status_code}")
                elif response.status_code != 200:
                    response.failure(f"unexpected: {response.status_code}")
                else:
                    response.success()

        @task
        def health(self) -> None:
            """Occasionally poll the health endpoint."""
            self.client.get("/api/ai/health", name="/api/ai/health")