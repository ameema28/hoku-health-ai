"""
Hoku Health Care - Performance Optimization Layer (Day 8).

Enforces strict time budgeting across the AI chatbot pipeline to guarantee
NFR-02 compliance (< 4 seconds total response time). Wraps each pipeline
stage with budget checks and timeout enforcement.

Key design decisions:
- MAX_TOTAL_TIME = 3.5s (leaving 0.5s buffer for network serialization)
- Per-stage budgets prevent any single step from monopolizing the pipeline
- generate_with_timeout provides universal timeout wrapper for any coroutine
- All budgets are logged when exceeded for operational visibility
"""

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Optional, Tuple, Type, Union

from app.ai.config import ai_settings

logger = logging.getLogger(__name__)

# Day 8.1: Exception classes that indicate a BUG IN OUR CODE, not a transient
# runtime failure. These must never be silently converted into a fallback
# value — doing so is what disguised
#     TypeError: Chain.invoke() missing 1 required positional argument: 'input'
# as an "LLM timeout" for an entire debugging session. Transport failures
# (network, HTTP, rate limits) still degrade gracefully; signature and
# attribute errors now surface loudly with a full traceback.
PROGRAMMING_ERRORS: Tuple[Type[BaseException], ...] = (
    TypeError,
    AttributeError,
    NameError,
    KeyError,
    IndexError,
)


class ResponseOptimizer:
    """
    Central performance optimizer for the Hoku AI chatbot pipeline.

    Tracks per-stage time budgets and enforces NFR-02 compliance by
    cutting off expensive operations before they breach the 4-second
    ceiling. Each stage has a dedicated budget; exceeding any budget
    triggers a warning log but does NOT halt execution — downstream
    stages must adapt to the remaining time.

    Attributes:
        MAX_TOTAL_TIME: Hard ceiling for the entire pipeline (seconds).
        TIME_BUDGETS: Per-stage allocation dictionary.
    """

    # NFR-02 ceiling: 3.5s pipeline + 0.5s network buffer = 4.0s total
    MAX_TOTAL_TIME: float = 3.5

    # Per-stage time allocations (sum of serial stages + concurrent max ≤ 3.5s)
    # REALISTIC budgets for Windows/SQLite cold-start development environment.
    TIME_BUDGETS: dict[str, float] = {
        "emergency_detect": 0.05,   # Regex: ~5ms actual
        "intent_classify": max(0.60, float(ai_settings.INTENT_CLASSIFICATION_TIMEOUT)),
        "memory_load": 0.60,        # SQLite cold-start: 300-500ms
        "rag_retrieve": 0.50,       # Embedding + similarity search
        "llm_generate": 1.40,       # Main LLM with compressed prompt
        "safety_check": 0.10,       # Regex validation: ~5-20ms
        "db_persist": 0.10,         # INSERT + COMMIT
        "doctor_lookup": 0.20,      # Symptom extract + DB query
    }

    # Stages executed concurrently via asyncio.gather. Their budgets must NOT
    # be summed against MAX_TOTAL_TIME — the wall-clock cost is max(), not sum().
    CONCURRENT_STAGES: frozenset = frozenset({
        "intent_classify", "memory_load", "rag_retrieve",
    })

    @classmethod
    def expected_serial_time(cls) -> float:
        """
        Wall-clock budget of the pipeline, accounting for concurrency.

        Use this instead of sum(TIME_BUDGETS.values()) when reasoning about
        NFR-02 headroom.
        """
        concurrent_cost = max(
            cls.TIME_BUDGETS[stage] for stage in cls.CONCURRENT_STAGES
        )
        sequential_cost = sum(
            budget for stage, budget in cls.TIME_BUDGETS.items()
            if stage not in cls.CONCURRENT_STAGES
        )
        return round(concurrent_cost + sequential_cost, 3)

    @classmethod
    def enforce_budget(cls, step_name: str, start_time: float) -> bool:
        """
        Check if the elapsed time for a pipeline step exceeds its budget.

        Logs a WARNING when the budget is exceeded so operators can
        identify slow stages. Returns False if over budget, True otherwise.
        This is advisory — the caller decides whether to abort or continue.

        Args:
            step_name: Key in TIME_BUDGETS (e.g., "intent_classify").
            start_time: time.perf_counter() captured before the step began.

        Returns:
            bool: True if within budget, False if exceeded.
        """
        elapsed = time.perf_counter() - start_time
        budget = cls.TIME_BUDGETS.get(step_name, 0.0)

        if elapsed > budget:
            logger.warning(
                "[PERFORMANCE] Step '%s' exceeded budget: %.3fs > %.3fs "
                "(over by %.3fs)",
                step_name,
                elapsed,
                budget,
                elapsed - budget,
            )
            return False

        logger.debug(
            "[PERFORMANCE] Step '%s' within budget: %.3fs <= %.3fs",
            step_name,
            elapsed,
            budget,
        )
        return True

    @classmethod
    def remaining_budget(cls, overall_start_time: float) -> float:
        """
        Calculate the remaining time left out of MAX_TOTAL_TIME.

        Used by the LLM generation stage to dynamically set its timeout
        so it never breaches the global ceiling, even if earlier stages
        ran over budget.

        Args:
            overall_start_time: time.perf_counter() captured at pipeline start.

        Returns:
            float: Seconds remaining (may be negative if already exceeded).
        """
        elapsed = time.perf_counter() - overall_start_time
        remaining = cls.MAX_TOTAL_TIME - elapsed
        return remaining

    @classmethod
    def get_budget_summary(cls, overall_start_time: float) -> dict[str, Any]:
        """
        Return a diagnostic summary of budget consumption.

        Useful for logging and metrics emission at the end of a request.

        Args:
            overall_start_time: Pipeline start timestamp.

        Returns:
            dict: Elapsed time, remaining budget, and budget status.
        """
        elapsed = time.perf_counter() - overall_start_time
        remaining = cls.MAX_TOTAL_TIME - elapsed
        return {
            "elapsed_seconds": round(elapsed, 3),
            "remaining_seconds": round(remaining, 3),
            "max_total_seconds": cls.MAX_TOTAL_TIME,
            "within_budget": remaining >= 0,
        }


async def generate_with_timeout(
    coro_or_func: Union[Coroutine[Any, Any, Any], Callable[..., Any]],
    timeout: float,
    fallback_value: Any = None,
    *args: Any,
    reraise_on: Tuple[Type[BaseException], ...] = PROGRAMMING_ERRORS,
    **kwargs: Any,
) -> Any:
    """
    Execute a coroutine or callable with a strict timeout, returning a
    fallback value on timeout instead of raising.

    This is the universal timeout wrapper used by every pipeline stage
    that calls an external service (Groq LLM, embedding model, etc.).
    It guarantees that no single call can monopolize the event loop
    and breach NFR-02.

    Args:
        coro_or_func: A coroutine object, an async function, or a sync callable.
        timeout: Maximum seconds to wait before returning fallback.
        fallback_value: Value returned if timeout or transient error occurs.
        *args: Positional arguments for callable (if not a coroutine).
        reraise_on: Exception classes treated as programming errors and
            propagated instead of converted to fallback_value. Defaults to
            PROGRAMMING_ERRORS. Pass an empty tuple () to restore the old
            swallow-everything behaviour.
        **kwargs: Keyword arguments for callable (if not a coroutine).

    IMPORTANT — argument forwarding contract:
        *args and **kwargs are forwarded VERBATIM to coro_or_func. A keyword
        whose name does not match the target's parameter name will land in the
        target's **kwargs and leave its positional parameters unbound. When in
        doubt, bind arguments at the call site with functools.partial and pass
        a zero-argument callable here.

    Returns:
        Any: Result of coro_or_func, or fallback_value on timeout/error.

    Saved execution time: ~200-500ms per call by cutting off slow LLM
    responses instead of waiting for the full 3.5s hard timeout.
    """
    try:
        if asyncio.iscoroutine(coro_or_func):
            result = await asyncio.wait_for(coro_or_func, timeout=timeout)
        elif asyncio.iscoroutinefunction(coro_or_func):
            result = await asyncio.wait_for(coro_or_func(*args, **kwargs), timeout=timeout)
        else:
            thread_task = asyncio.to_thread(coro_or_func, *args, **kwargs)
            result = await asyncio.wait_for(thread_task, timeout=timeout)
        return result

    except asyncio.TimeoutError:
        logger.warning(
            "[PERFORMANCE] Operation timed out after %.3fs — returning fallback",
            timeout,
        )
        return fallback_value

    except reraise_on:
        logger.exception(
            "[PERFORMANCE] Programming error inside timed operation — re-raising"
        )
        raise

    except Exception as exc:
        logger.warning(
            "[PERFORMANCE] Operation failed: %s — returning fallback",
            exc,
        )
        return fallback_value