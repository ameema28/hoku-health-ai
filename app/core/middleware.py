"""
Hoku Health Care - Custom Middleware.

Request timing and NFR compliance monitoring for critical endpoints.
"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request latency and alerts on NFR breaches.

    Specifically monitors /api/ai/chat to ensure the <4s response time
    requirement (NFR-02) is enforced and violations are logged.
    """

    # Endpoints subject to strict timing requirements
    TIMED_ENDPOINTS = {"/api/ai/chat"}

    # NFR-02: Maximum allowed response time in seconds
    NFR_02_LIMIT_SECONDS: float = 4.0

    # Alert threshold: log warning if over 80% of limit
    WARNING_THRESHOLD: float = 0.8 * NFR_02_LIMIT_SECONDS  # 3.2s

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Time the request and log latency metrics.

        Args:
            request: Incoming FastAPI request.
            call_next: Next middleware/endpoint in the stack.

        Returns:
            Response: The HTTP response from downstream handlers.
        """
        start_time = time.perf_counter()
        path = request.url.path

        response = await call_next(request)

        elapsed = time.perf_counter() - start_time

        # Only log timing for monitored endpoints
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

            # Critical alert if NFR breached
            if elapsed > self.NFR_02_LIMIT_SECONDS:
                logger.error(
                    "NFR-02 BREACH: %s took %.3fs (limit: %.3fs). "
                    "User may have experienced degraded service.",
                    path,
                    elapsed,
                    self.NFR_02_LIMIT_SECONDS,
                )

        return response