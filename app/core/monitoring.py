"""
Hoku Health Care - Safety & Performance Monitoring Layer.

Day 7: Thread-safe in-memory metrics collection for:
- Emergency detection counters
- Safety violation counters
- NFR-02 latency tracking (< 4s ceiling)

Day 10: Prometheus exposition on top of the same collector. The in-memory
``HokuMetrics`` API is unchanged - every existing call site and all 215
existing tests keep working - and each mutation now *also* updates a
Prometheus collector when ``prometheus-client`` is installed.

Exported series (all registered on the private ``HOKU_REGISTRY``):

===========================================  =========  ==========================
Metric                                       Type       Labels
===========================================  =========  ==========================
hoku_chatbot_requests_total                  Counter    intent, emergency_flag
hoku_chatbot_response_time_seconds           Histogram  endpoint
hoku_chatbot_safety_violations_total         Counter    violation_type
hoku_chatbot_emergency_escalations_total     Counter    urgency
hoku_chatbot_cache_hit_ratio                 Gauge      -
hoku_chatbot_nfr02_breaches_total            Counter    endpoint
hoku_chatbot_safety_fallbacks_total          Counter    -
hoku_chatbot_build_info                      Info       version, environment
===========================================  =========  ==========================

A private ``CollectorRegistry`` is used instead of the global default so
that repeated imports under pytest cannot raise ``Duplicated timeseries``.

Cache-ratio integration
-----------------------
``ResponseCache`` (app/ai/caching.py) does not itself count hits and
misses. To populate ``hoku_chatbot_cache_hit_ratio``, add one line at
each of the three cache branches already present in
``app/ai/chatbot.py::get_response``::

    metrics.record_cache_hit()    # fast-path HIT, and full-lookup HIT
    metrics.record_cache_miss()   # after the "Cache MISS" debug log

Until those calls exist the gauge simply reports 0.0; nothing else in
the pipeline is affected.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Day 10: optional Prometheus dependency
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    CollectorRegistry = None  # type: ignore[assignment,misc]
    Counter = Gauge = Histogram = Info = None  # type: ignore[assignment,misc]
    generate_latest = None  # type: ignore[assignment]
    logger.warning(
        "prometheus-client not installed; /metrics will report a stub. "
        "Install it with: pip install prometheus-client==0.20.0"
    )


# Private registry - never the global default, so re-imports under pytest
# cannot collide on duplicate timeseries names.
HOKU_REGISTRY: Any = CollectorRegistry() if PROMETHEUS_AVAILABLE else None

# NFR-02 aware bucket boundaries. The 4.0 bucket is the SLO edge, so
# `hoku_chatbot_response_time_seconds_bucket{le="4.0"}` divided by
# `_count` is the compliance ratio straight out of Grafana.
_LATENCY_BUCKETS: Tuple[float, ...] = (
    0.005, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 6.0, 10.0,
)

if PROMETHEUS_AVAILABLE:
    PROM_REQUESTS_TOTAL = Counter(
        "hoku_chatbot_requests_total",
        "Total chat requests processed, by classified intent and emergency flag.",
        labelnames=("intent", "emergency_flag"),
        registry=HOKU_REGISTRY,
    )
    PROM_RESPONSE_TIME = Histogram(
        "hoku_chatbot_response_time_seconds",
        "End-to-end chatbot response latency in seconds (NFR-02 ceiling: 4s).",
        labelnames=("endpoint",),
        buckets=_LATENCY_BUCKETS,
        registry=HOKU_REGISTRY,
    )
    PROM_SAFETY_VIOLATIONS = Counter(
        "hoku_chatbot_safety_violations_total",
        "Post-LLM clinical safety violations blocked by the guardrails.",
        labelnames=("violation_type",),
        registry=HOKU_REGISTRY,
    )
    PROM_EMERGENCY_ESCALATIONS = Counter(
        "hoku_chatbot_emergency_escalations_total",
        "Emergency detections that short-circuited the LLM pipeline.",
        labelnames=("urgency",),
        registry=HOKU_REGISTRY,
    )
    PROM_CACHE_HIT_RATIO = Gauge(
        "hoku_chatbot_cache_hit_ratio",
        "Response cache hit ratio (0.0-1.0). EMERGENCY and SYMPTOM "
        "intents are never cached, so they never contribute.",
        registry=HOKU_REGISTRY,
    )
    PROM_NFR02_BREACHES = Counter(
        "hoku_chatbot_nfr02_breaches_total",
        "Requests that exceeded the NFR-02 4-second response ceiling.",
        labelnames=("endpoint",),
        registry=HOKU_REGISTRY,
    )
    PROM_SAFETY_FALLBACKS = Counter(
        "hoku_chatbot_safety_fallbacks_total",
        "3-strike safety fallbacks served instead of an LLM response.",
        registry=HOKU_REGISTRY,
    )
    PROM_BUILD_INFO = Info(
        "hoku_chatbot_build",
        "Static build and environment metadata.",
        registry=HOKU_REGISTRY,
    )
else:  # pragma: no cover
    PROM_REQUESTS_TOTAL = None  # type: ignore[assignment]
    PROM_RESPONSE_TIME = None  # type: ignore[assignment]
    PROM_SAFETY_VIOLATIONS = None  # type: ignore[assignment]
    PROM_EMERGENCY_ESCALATIONS = None  # type: ignore[assignment]
    PROM_CACHE_HIT_RATIO = None  # type: ignore[assignment]
    PROM_NFR02_BREACHES = None  # type: ignore[assignment]
    PROM_SAFETY_FALLBACKS = None  # type: ignore[assignment]
    PROM_BUILD_INFO = None  # type: ignore[assignment]


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

    Day 10: every mutating method additionally updates the Prometheus
    collectors declared above, when prometheus-client is installed.

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

        # Day 10: cache observability
        self._cache_hits_total: int = 0
        self._cache_misses_total: int = 0

        # Latency history (circular buffer, last 1000 measurements)
        self._latency_history: List[LatencySnapshot] = []
        self._max_latency_history: int = 1000

    # ------------------------------------------------------------------
    # Counter Operations
    # ------------------------------------------------------------------

    def increment_emergency_detection(self, urgency: str = "high") -> None:
        """
        Increment the total emergency detection counter.

        Args:
            urgency: Detected urgency tier ("high" or "moderate"). Used
                only as a Prometheus label; the in-memory counter stays
                a single scalar for backwards compatibility.
        """
        with self._counter_lock:
            self._emergency_detections_total += 1
        if PROM_EMERGENCY_ESCALATIONS is not None:
            PROM_EMERGENCY_ESCALATIONS.labels(urgency=str(urgency or "high")).inc()
        logger.debug(
            "Emergency detection counter incremented to %d",
            self._emergency_detections_total,
        )

    def increment_safety_violation(self, violation_type: Optional[str] = None) -> None:
        """
        Increment the total safety violation counter.

        Args:
            violation_type: Optional violation type for detailed logging
                and as the Prometheus label value.
        """
        with self._counter_lock:
            self._safety_violations_total += 1
        if PROM_SAFETY_VIOLATIONS is not None:
            PROM_SAFETY_VIOLATIONS.labels(
                violation_type=str(violation_type or "unknown")
            ).inc()
        logger.debug(
            "Safety violation counter incremented to %d (type=%s)",
            self._safety_violations_total,
            violation_type or "unknown",
        )

    def increment_3_strike_fallback(self) -> None:
        """Increment the 3-strike safety fallback counter."""
        with self._counter_lock:
            self._safety_3_strike_fallbacks_total += 1
        if PROM_SAFETY_FALLBACKS is not None:
            PROM_SAFETY_FALLBACKS.inc()
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
        if PROM_NFR02_BREACHES is not None:
            PROM_NFR02_BREACHES.labels(endpoint=endpoint).inc()
        logger.error(
            "NFR-02 breach counter incremented to %d (endpoint=%s)",
            self._nfr02_breaches_total,
            endpoint,
        )

    def increment_request(
        self,
        endpoint: str = "/api/ai/chat",
        intent: str = "unknown",
        emergency_flag: bool = False,
    ) -> None:
        """
        Increment the total request counter.

        The extra keyword arguments are additive (Day 10) - existing
        single-argument call sites in ``app/ai/chatbot.py`` continue to
        work unchanged and simply land in the ``intent="unknown"`` series.

        Args:
            endpoint: The endpoint that received the request.
            intent: Classified intent, when known at call time.
            emergency_flag: Whether emergency detection fired.
        """
        with self._counter_lock:
            self._requests_total += 1
        if PROM_REQUESTS_TOTAL is not None:
            PROM_REQUESTS_TOTAL.labels(
                intent=str(intent or "unknown"),
                emergency_flag="true" if emergency_flag else "false",
            ).inc()
        logger.debug(
            "Request counter incremented to %d (endpoint=%s)",
            self._requests_total,
            endpoint,
        )

    # ------------------------------------------------------------------
    # Day 10: Cache observability
    # ------------------------------------------------------------------

    def record_cache_hit(self) -> None:
        """Record a response-cache hit and refresh the ratio gauge."""
        with self._counter_lock:
            self._cache_hits_total += 1
        self._sync_cache_gauge()

    def record_cache_miss(self) -> None:
        """Record a response-cache miss and refresh the ratio gauge."""
        with self._counter_lock:
            self._cache_misses_total += 1
        self._sync_cache_gauge()

    def get_cache_hit_ratio(self) -> float:
        """
        Return the cache hit ratio.

        Returns:
            float: hits / (hits + misses), or 0.0 when no lookups have
            been recorded yet.
        """
        with self._counter_lock:
            total = self._cache_hits_total + self._cache_misses_total
            if total == 0:
                return 0.0
            return self._cache_hits_total / total

    def _sync_cache_gauge(self) -> None:
        """Push the current hit ratio into the Prometheus gauge."""
        if PROM_CACHE_HIT_RATIO is not None:
            PROM_CACHE_HIT_RATIO.set(self.get_cache_hit_ratio())

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

        if PROM_RESPONSE_TIME is not None:
            PROM_RESPONSE_TIME.labels(endpoint=endpoint).observe(elapsed_seconds)

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

    def get_cache_hits_total(self) -> int:
        """Return total response-cache hits (Day 10)."""
        with self._counter_lock:
            return self._cache_hits_total

    def get_cache_misses_total(self) -> int:
        """Return total response-cache misses (Day 10)."""
        with self._counter_lock:
            return self._cache_misses_total

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
                measurements = [
                    s.elapsed_ms for s in self._latency_history if s.endpoint == endpoint
                ]
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
            float: Percentage of requests that breached NFR-02 (0.0-100.0).
        """
        with self._counter_lock:
            if self._requests_total == 0:
                return 0.0
            return (self._nfr02_breaches_total / self._requests_total) * 100.0

    def get_summary(self) -> Dict[str, Any]:
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
            # Day 10 additions
            "cache_hits_total": self.get_cache_hits_total(),
            "cache_misses_total": self.get_cache_misses_total(),
            "cache_hit_ratio": round(self.get_cache_hit_ratio(), 4),
            "prometheus_enabled": PROMETHEUS_AVAILABLE,
        }

    def reset_counters(self) -> None:
        """Reset all counters to zero (useful for testing)."""
        with self._counter_lock:
            self._emergency_detections_total = 0
            self._safety_violations_total = 0
            self._safety_3_strike_fallbacks_total = 0
            self._nfr02_breaches_total = 0
            self._requests_total = 0
            self._cache_hits_total = 0
            self._cache_misses_total = 0
        with self._latency_lock:
            self._latency_history.clear()
        self._sync_cache_gauge()
        logger.info("All metrics counters reset")


def get_metrics() -> HokuMetrics:
    """
    Factory function returning the singleton HokuMetrics instance.

    Returns:
        HokuMetrics: The shared thread-safe metrics collector.
    """
    return HokuMetrics()


# ---------------------------------------------------------------------------
# Day 10: Prometheus exposition helpers
# ---------------------------------------------------------------------------
def set_build_info(version: str, environment: str) -> None:
    """
    Publish static build metadata as ``hoku_chatbot_build_info``.

    Called once from the application lifespan in ``app/main.py``.

    Args:
        version: Application version string (e.g. "1.0.0").
        environment: Deployment environment (e.g. "production").
    """
    if PROM_BUILD_INFO is not None:
        PROM_BUILD_INFO.info({"version": version, "environment": environment})


def render_prometheus() -> Tuple[bytes, str]:
    """
    Render the current registry in Prometheus text exposition format.

    Returns:
        Tuple[bytes, str]: The payload and its Content-Type header value.
        When prometheus-client is not installed, a single ``# stub``
        comment is returned so scrapes fail soft rather than 500.
    """
    if not PROMETHEUS_AVAILABLE or generate_latest is None:
        return (
            b"# prometheus-client is not installed; metrics unavailable\n",
            CONTENT_TYPE_LATEST,
        )
    # Refresh derived gauges immediately before scraping.
    get_metrics()._sync_cache_gauge()
    return generate_latest(HOKU_REGISTRY), CONTENT_TYPE_LATEST