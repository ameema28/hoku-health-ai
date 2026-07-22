"""
Hoku Health Care - Custom Middleware (Day 8: Enhanced Performance Monitoring).

Request timing, NFR-02 compliance monitoring, and performance statistics
tracking for critical endpoints. Adds X-Response-Time-Sec header to all
responses and maintains thread-safe performance metrics.

Day 8 additions:
- PerformanceMetricsStore: Thread-safe tracker for latency samples,
  SLA violations, and rolling statistics.
- X-Response-Time-Sec header on every response.
- ERROR-level logging for NFR-02 breaches with standardized format.
- get_stats() helper for admin monitoring endpoint integration.
"""

import logging
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class PerformanceMetricsStore:
    """
    Thread-safe in-memory store for request latency metrics.

    Maintains a rolling window of recent latency samples and tracks
    SLA violations for operational visibility. Designed to be
    lightweight — no external dependencies.

    Attributes:
        MAX_SAMPLES: Maximum number of recent latency samples to retain.
        SLA_LIMIT_SECONDS: NFR-02 response time ceiling.
    """

    MAX_SAMPLES: int = 1000
    SLA_LIMIT_SECONDS: float = 4.0

    def __init__(self) -> None:
        """Initialize thread-safe storage."""
        self._lock = threading.Lock()
        self._samples: deque = deque(maxlen=self.MAX_SAMPLES)
        self._violation_count: int = 0
        self._total_requests: int = 0

    def record(self, path: str, elapsed_seconds: float, status_code: int) -> None:
        """
        Record a single request's latency metrics.

        Args:
            path: Request path.
            elapsed_seconds: Total request duration.
            status_code: HTTP status code.
        """
        with self._lock:
            self._total_requests += 1
            is_violation = elapsed_seconds > self.SLA_LIMIT_SECONDS
            if is_violation:
                self._violation_count += 1

            self._samples.append({
                "path": path,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "status_code": status_code,
                "timestamp": time.time(),
                "breached": is_violation,
            })

    def get_stats(self) -> Dict[str, any]:
        """
        Return performance statistics for admin monitoring.

        Returns:
            dict: Summary of latency distribution, violation counts,
                  and recent sample statistics.
        """
        with self._lock:
            if not self._samples:
                return {
                    "total_requests": 0,
                    "sla_violations": 0,
                    "breach_rate_percent": 0.0,
                    "average_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                    "recent_samples": [],
                }

            latencies = [s["elapsed_seconds"] for s in self._samples]
            sorted_latencies = sorted(latencies)
            p99_index = int(len(sorted_latencies) * 0.99)
            p99 = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]

            breach_rate = (
                (self._violation_count / self._total_requests) * 100.0
                if self._total_requests > 0
                else 0.0
            )

            return {
                "total_requests": self._total_requests,
                "sla_violations": self._violation_count,
                "breach_rate_percent": round(breach_rate, 2),
                "average_latency_ms": round(sum(latencies) / len(latencies) * 1000, 2),
                "p99_latency_ms": round(p99 * 1000, 2),
                "recent_samples": list(self._samples)[-10:],  # Last 10 for detail
            }

    def reset(self) -> None:
        """Clear all metrics (useful for testing)."""
        with self._lock:
            self._samples.clear()
            self._violation_count = 0
            self._total_requests = 0


# Module-level singleton — shared across all requests
_metrics_store = PerformanceMetricsStore()


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request latency, alerts on NFR breaches,
    and adds timing headers to all responses.

    Day 8 enhancements:
    - X-Response-Time-Sec header on every response
    - ERROR-level logging for NFR-02 breaches with [NFR-02 EXCEEDED] tag
    - Integration with PerformanceMetricsStore for rolling statistics
    """

    # Endpoints subject to strict timing requirements
    TIMED_ENDPOINTS: set[str] = {"/api/ai/chat"}

    # NFR-02: Maximum allowed response time in seconds
    NFR_02_LIMIT_SECONDS: float = 4.0

    # Alert threshold: log warning if over 80% of limit
    WARNING_THRESHOLD: float = 0.8 * NFR_02_LIMIT_SECONDS  # 3.2s

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Time the request, add X-Response-Time-Sec header, and log metrics.

        Args:
            request: Incoming FastAPI request.
            call_next: Next middleware/endpoint in the stack.

        Returns:
            Response: The HTTP response with timing headers attached.
        """
        start_time = time.perf_counter()
        path = request.url.path

        response = await call_next(request)

        elapsed = time.perf_counter() - start_time

        # ------------------------------------------------------------------
        # Day 8: Add X-Response-Time-Sec header to ALL responses
        # Saved execution time: N/A (observability only, < 0.1ms overhead)
        # ------------------------------------------------------------------
        response.headers["X-Response-Time-Sec"] = f"{elapsed:.4f}"

        # Record in metrics store for all paths (not just timed endpoints)
        _metrics_store.record(path, elapsed, response.status_code)

        # Only log detailed timing for monitored endpoints
        if path in self.TIMED_ENDPOINTS:
            status_code = response.status_code

            # Always log timing for AI chat endpoint
            logger.info(
                "TIMING %s %s — status=%d, latency=%.3fs",
                request.method,
                path,
                status_code,
                elapsed,
            )

            # Warning if approaching limit
            if elapsed > self.WARNING_THRESHOLD:
                logger.warning(
                    "LATENCY WARNING: %s took %.3fs (threshold: %.3fs)",
                    path,
                    elapsed,
                    self.WARNING_THRESHOLD,
                )

            # ------------------------------------------------------------------
            # Day 8: ERROR-level logging for NFR-02 breach
            # Standardized format for alerting infrastructure parsing
            # ------------------------------------------------------------------
            if elapsed > self.NFR_02_LIMIT_SECONDS:
                logger.error(
                    "[ERROR] [NFR-02 EXCEEDED] %s %s took %.4fs "
                    "(limit: %.3fs). User may have experienced degraded service.",
                    request.method,
                    path,
                    elapsed,
                    self.NFR_02_LIMIT_SECONDS,
                )

        return response

    @classmethod
    def get_stats(cls) -> Dict[str, any]:
        """
        Expose performance statistics for admin monitoring.

        Returns:
            dict: Rolling latency statistics and SLA violation counts.
        """
        return _metrics_store.get_stats()

    @classmethod
    def reset_stats(cls) -> None:
        """Reset all performance statistics (useful for testing)."""
        _metrics_store.reset()