"""
Hoku Health Care - Per-User Rate Limiting (Day 10).

A dependency-free, thread-safe, in-process sliding-window limiter used to
cap ``POST /api/ai/chat`` at N requests per minute per authenticated user
(default 5, from ``settings.RATE_LIMIT_REQUESTS_PER_MINUTE``).

Why in-process rather than Redis/slowapi
---------------------------------------
The rest of the observability stack (``HokuMetrics``, ``ResponseCache``)
is already an in-process singleton and the service is deployed as a
single uvicorn worker per container (see the Dockerfile CMD). A local
sliding window keeps the dependency surface minimal and adds well under a
millisecond per request. When the platform scales to multiple workers or
replicas, swap :func:`get_chat_rate_limiter` for a Redis-backed
implementation with the same ``check()`` signature - no call-site change.

Algorithm
---------
Per key, a ``deque`` of request timestamps. On ``check()`` we evict
timestamps older than the window, and if the survivor count is below the
limit we admit the request and record ``now``; otherwise we compute how
long until the oldest timestamp ages out and raise
:class:`RateLimitExceeded` with that ``retry_after``.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional


class RateLimitExceeded(Exception):
    """
    Raised when a key has exhausted its request quota for the window.

    Attributes:
        retry_after_seconds: Whole seconds until the next request is allowed.
        limit: The configured request ceiling for the window.
    """

    def __init__(self, retry_after_seconds: int, limit: int) -> None:
        """
        Initialize the exception.

        Args:
            retry_after_seconds: Seconds until quota frees up (min 1).
            limit: The window's request ceiling.
        """
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.limit = limit
        super().__init__(
            f"Rate limit of {limit} requests exceeded; "
            f"retry after {self.retry_after_seconds}s"
        )


@dataclass
class RateLimitState:
    """
    Snapshot returned when a request is admitted.

    Attributes:
        limit: The window's request ceiling.
        remaining: Requests still available in the current window.
        reset_seconds: Seconds until the window fully clears.
    """

    limit: int
    remaining: int
    reset_seconds: int


class SlidingWindowRateLimiter:
    """
    Thread-safe fixed-duration sliding-window limiter.

    One instance guards one logical resource (here, the chat endpoint).
    Keys are arbitrary strings; the caller supplies the authenticated
    user id.

    Attributes:
        limit: Maximum requests permitted per window.
        window_seconds: Rolling window length in seconds.
    """

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        """
        Initialize the limiter.

        Args:
            limit: Maximum requests per window (must be >= 1).
            window_seconds: Window length in seconds.
        """
        self.limit = max(1, int(limit))
        self.window_seconds = float(window_seconds)
        self._lock = threading.Lock()
        self._hits: Dict[str, Deque[float]] = {}

    def check(self, key: str, now: Optional[float] = None) -> RateLimitState:
        """
        Admit or reject a single request for ``key``.

        Args:
            key: The rate-limit key (typically ``str(user_id)``).
            now: Optional monotonic-ish timestamp override for tests.

        Returns:
            RateLimitState: Remaining quota when the request is admitted.

        Raises:
            RateLimitExceeded: When the key is over its quota.
        """
        current = time.time() if now is None else now
        cutoff = current - self.window_seconds

        with self._lock:
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket

            # Evict timestamps that have aged out of the window.
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.limit:
                oldest = bucket[0]
                retry_after = (oldest + self.window_seconds) - current
                raise RateLimitExceeded(
                    retry_after_seconds=int(retry_after) + 1,
                    limit=self.limit,
                )

            bucket.append(current)
            remaining = self.limit - len(bucket)
            reset = int((bucket[0] + self.window_seconds) - current) + 1
            return RateLimitState(
                limit=self.limit,
                remaining=remaining,
                reset_seconds=max(1, reset),
            )

    def reset(self, key: Optional[str] = None) -> None:
        """
        Clear recorded hits.

        Args:
            key: A specific key to clear, or None to clear everything.
                Primarily used by tests to isolate cases.
        """
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_chat_rate_limiter: Optional[SlidingWindowRateLimiter] = None
_limiter_lock = threading.Lock()


def get_chat_rate_limiter() -> SlidingWindowRateLimiter:
    """
    Return the process-wide limiter for the chat endpoint.

    Lazily constructed from ``settings.RATE_LIMIT_REQUESTS_PER_MINUTE`` on
    first use so that test environments importing this module do not pay
    for it unless they call the endpoint.

    Returns:
        SlidingWindowRateLimiter: The shared limiter instance.
    """
    global _chat_rate_limiter
    if _chat_rate_limiter is None:
        with _limiter_lock:
            if _chat_rate_limiter is None:
                from app.core.config import settings

                _chat_rate_limiter = SlidingWindowRateLimiter(
                    limit=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                    window_seconds=60.0,
                )
    return _chat_rate_limiter


def reset_chat_rate_limiter() -> None:
    """Reset the chat limiter's state (test helper)."""
    if _chat_rate_limiter is not None:
        _chat_rate_limiter.reset()