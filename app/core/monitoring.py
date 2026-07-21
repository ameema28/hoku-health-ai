"""
Hoku Health Care - Safety & Performance Monitoring Layer (Day 7).

Thread-safe in-memory metrics collection for:
- Emergency detection counters
- Safety violation counters
- NFR-02 latency tracking (< 4s ceiling)

Designed for lightweight observability without external dependencies.
Can be extended to push to Prometheus/Grafana in production.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LatencySnapshot:
    """
    A single latency measurement snapshot.

    Attributes:
        endpoint: The API endpoint path.
        elapsed_ms: Request duration in milliseconds.
        timestamp: Unix timestamp of the measurement.
        breached: Whether the measurement exceeded NFR-02 (4s).
    """
    endpoint: str
    elapsed_ms: float
    timestamp: float = field(default_factory=time.time)
    breached: bool = False


class HokuMetrics:
    """
    Thread-safe in-memory metrics collector for Hoku Health Care.

    Tracks counters and latency distributions for safety monitoring
    and NFR-02 compliance reporting. All operations are thread-safe
    via internal locking.

    Singleton pattern: use `get_metrics()` to access the shared instance.
    """

    _instance: Optional["HokuMetrics"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "HokuMetrics":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_storage()
        return cls._instance

    def _init_storage(self) -> None:
        """Initialize internal counters and storage."""
        self._counter_lock = threading.Lock()
        self._latency_lock = threading.Lock()

        # Counters
        self._emergency_detections_total: int = 0
        self._safety_violations_total: int = 0
        self._safety_3_strike_fallbacks_total: int = 0
        self._nfr02_breaches_total: int = 0
        self._requests_total: int = 0

        # Latency history (circular buffer, last 1000 measurements)
        self._latency_history: List[LatencySnapshot] = []
        self._max_latency_history: int = 1000

    # ------------------------------------------------------------------
    # Counter Operations
    # ------------------------------------------------------------------

    def increment_emergency_detection(self) -> None:
        """Increment the total emergency detection counter."""
        with self._counter_lock:
            self._emergency_detections_total += 1
        logger.debug("Emergency detection counter incremented to %d", self._emergency_detections_total)

    def increment_safety_violation(self, violation_type: Optional[str] = None) -> None:
        """
        Increment the total safety violation counter.

        Args:
            violation_type: Optional violation type for detailed logging.
        """
        with self._counter_lock:
            self._safety_violations_total += 1
        logger.debug(
            "Safety violation counter incremented to %d (type=%s)",
            self._safety_violations_total,
            violation_type or "unknown",
        )

    def increment_3_strike_fallback(self) -> None:
        """Increment the 3-strike safety fallback counter."""
        with self._counter_lock:
            self._safety_3_strike_fallbacks_total += 1
        logger.critical(
            "3-strike fallback counter incremented to %d",
            self._safety_3_strike_fallbacks_total,
        )

    def increment_nfr02_breach(self, endpoint: str = "/api/ai/chat") -> None:
        """
        Increment the NFR-02 breach counter.

        Args:
            endpoint: The endpoint that breached the latency requirement.
        """
        with self._counter_lock:
            self._nfr02_breaches_total += 1
        logger.error(
            "NFR-02 breach counter incremented to %d (endpoint=%s)",
            self._nfr02_breaches_total,
            endpoint,
        )

    def increment_request(self, endpoint: str = "/api/ai/chat") -> None:
        """
        Increment the total request counter.

        Args:
            endpoint: The endpoint that received the request.
        """
        with self._counter_lock:
            self._requests_total += 1
        logger.debug(
            "Request counter incremented to %d (endpoint=%s)",
            self._requests_total,
            endpoint,
        )

    # ------------------------------------------------------------------
    # Latency Tracking
    # ------------------------------------------------------------------

    def record_latency(
        self,
        endpoint: str,
        elapsed_seconds: float,
        nfr_limit_seconds: float = 4.0,
    ) -> None:
        """
        Record a latency measurement and check for NFR-02 breach.

        Args:
            endpoint: The API endpoint path.
            elapsed_seconds: Request duration in seconds.
            nfr_limit_seconds: The NFR-02 latency limit (default 4.0s).
        """
        elapsed_ms = elapsed_seconds * 1000.0
        breached = elapsed_seconds > nfr_limit_seconds

        snapshot = LatencySnapshot(
            endpoint=endpoint,
            elapsed_ms=elapsed_ms,
            breached=breached,
        )

        with self._latency_lock:
            self._latency_history.append(snapshot)
            # Trim to max size (circular buffer behavior)
            if len(self._latency_history) > self._max_latency_history:
                self._latency_history = self._latency_history[-self._max_latency_history:]

        if breached:
            self.increment_nfr02_breach(endpoint)

        logger.info(
            "Latency recorded: endpoint=%s, elapsed=%.3fms, breached=%s",
            endpoint,
            elapsed_ms,
            breached,
        )

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def get_emergency_detections_total(self) -> int:
        """Return total emergency detections."""
        with self._counter_lock:
            return self._emergency_detections_total

    def get_safety_violations_total(self) -> int:
        """Return total safety violations."""
        with self._counter_lock:
            return self._safety_violations_total

    def get_3_strike_fallbacks_total(self) -> int:
        """Return total 3-strike fallback events."""
        with self._counter_lock:
            return self._safety_3_strike_fallbacks_total

    def get_nfr02_breaches_total(self) -> int:
        """Return total NFR-02 latency breaches."""
        with self._counter_lock:
            return self._nfr02_breaches_total

    def get_requests_total(self) -> int:
        """Return total requests processed."""
        with self._counter_lock:
            return self._requests_total

    def get_average_latency_ms(self, endpoint: Optional[str] = None) -> float:
        """
        Calculate average latency in milliseconds.

        Args:
            endpoint: Optional endpoint filter. If None, averages across all.

        Returns:
            float: Average latency in ms, or 0.0 if no measurements.
        """
        with self._latency_lock:
            if not self._latency_history:
                return 0.0

            if endpoint:
                measurements = [s for s in self._latency_history if s.endpoint == endpoint]
            else:
                measurements = self._latency_history

            if not measurements:
                return 0.0

            return sum(s.elapsed_ms for s in measurements) / len(measurements)

    def get_p99_latency_ms(self, endpoint: Optional[str] = None) -> float:
        """
        Calculate P99 latency in milliseconds.

        Args:
            endpoint: Optional endpoint filter.

        Returns:
            float: P99 latency in ms, or 0.0 if no measurements.
        """
        with self._latency_lock:
            if not self._latency_history:
                return 0.0

            if endpoint:
                measurements = [s.elapsed_ms for s in self._latency_history if s.endpoint == endpoint]
            else:
                measurements = [s.elapsed_ms for s in self._latency_history]

            if not measurements:
                return 0.0

            sorted_ms = sorted(measurements)
            p99_index = int(len(sorted_ms) * 0.99)
            return sorted_ms[min(p99_index, len(sorted_ms) - 1)]

    def get_breach_rate(self) -> float:
        """
        Calculate the NFR-02 breach rate as a percentage.

        Returns:
            float: Percentage of requests that breached NFR-02 (0.0–100.0).
        """
        with self._counter_lock:
            if self._requests_total == 0:
                return 0.0
            return (self._nfr02_breaches_total / self._requests_total) * 100.0

    def get_summary(self) -> Dict[str, any]:
        """
        Return a complete metrics summary dictionary.

        Returns:
            Dict: All counters, averages, and breach rates.
        """
        return {
            "emergency_detections_total": self.get_emergency_detections_total(),
            "safety_violations_total": self.get_safety_violations_total(),
            "safety_3_strike_fallbacks_total": self.get_3_strike_fallbacks_total(),
            "nfr02_breaches_total": self.get_nfr02_breaches_total(),
            "requests_total": self.get_requests_total(),
            "average_latency_ms": self.get_average_latency_ms("/api/ai/chat"),
            "p99_latency_ms": self.get_p99_latency_ms("/api/ai/chat"),
            "breach_rate_percent": round(self.get_breach_rate(), 2),
        }

    def reset_counters(self) -> None:
        """Reset all counters to zero (useful for testing)."""
        with self._counter_lock:
            self._emergency_detections_total = 0
            self._safety_violations_total = 0
            self._safety_3_strike_fallbacks_total = 0
            self._nfr02_breaches_total = 0
            self._requests_total = 0
        with self._latency_lock:
            self._latency_history.clear()
        logger.info("All metrics counters reset")


def get_metrics() -> HokuMetrics:
    """
    Factory function returning the singleton HokuMetrics instance.

    Returns:
        HokuMetrics: The shared thread-safe metrics collector.
    """
    return HokuMetrics()