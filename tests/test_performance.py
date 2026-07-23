"""
Hoku Health Care - Performance Optimization Unit Tests (Day 8).

Comprehensive tests for the performance layer:
- ResponseOptimizer budget enforcement
- generate_with_timeout timeout handling
- ResponseCache hit/miss performance and safety rules
- LLMFactory model selection
- compress_prompt context compression

All external API calls are mocked — no real Groq usage.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from app.ai.caching import ResponseCache
from app.ai.fallback_responses import (
    FALLBACK_BOOKING,
    FALLBACK_EMERGENCY,
    FALLBACK_GENERAL,
    get_fallback_for_intent,
)
from app.ai.llm_optimizer import LLMFactory
from app.ai.ai_performance import ResponseOptimizer, generate_with_timeout


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def response_cache():
    """Provide a fresh ResponseCache instance for each test."""
    cache = ResponseCache()
    cache.clear()
    yield cache
    cache.clear()


@pytest.fixture
def optimizer():
    """Provide ResponseOptimizer class access."""
    return ResponseOptimizer


# ------------------------------------------------------------------
# ResponseOptimizer Tests
# ------------------------------------------------------------------

class TestResponseOptimizer:

    def test_max_total_time_constant(self, optimizer):
        """MAX_TOTAL_TIME must be 3.5s for NFR-02 compliance."""
        assert optimizer.MAX_TOTAL_TIME == 3.5

    def test_time_budgets_sum_to_max(self, optimizer):
        """Concurrent-aware serial time must fit within MAX_TOTAL_TIME."""
        # Day 8.1: intent_classify, memory_load, and rag_retrieve run
        # concurrently, so their wall-clock cost is max(), not sum().
        # expected_serial_time() accounts for this overlap.
        total_budget = optimizer.expected_serial_time()
        assert total_budget <= optimizer.MAX_TOTAL_TIME

    def test_enforce_budget_within_limit(self, optimizer):
        """Budget check passes when within allocated time."""
        start = time.perf_counter()
        # Simulate a fast operation
        time.sleep(0.001)
        result = optimizer.enforce_budget("emergency_detect", start)
        assert result is True  # 1ms < 50ms budget

    def test_enforce_budget_exceeded(self, optimizer):
        """Budget check fails and logs warning when exceeded."""
        start = time.perf_counter() - 1.0  # Pretend we started 1 second ago
        result = optimizer.enforce_budget("emergency_detect", start)
        # 1.0s elapsed > 0.05s budget → should return False
        assert result is False

    def test_remaining_budget_positive(self, optimizer):
        """Remaining budget is positive shortly after start."""
        start = time.perf_counter()
        time.sleep(0.01)
        remaining = optimizer.remaining_budget(start)
        assert remaining > 0
        assert remaining < optimizer.MAX_TOTAL_TIME

    def test_remaining_budget_negative_when_exceeded(self, optimizer):
        """Remaining budget is negative when total time exceeded."""
        start = time.perf_counter() - 5.0  # Started 5 seconds ago
        remaining = optimizer.remaining_budget(start)
        assert remaining < 0

    def test_get_budget_summary(self, optimizer):
        """Budget summary contains expected keys."""
        start = time.perf_counter()
        time.sleep(0.01)
        summary = optimizer.get_budget_summary(start)
        assert "elapsed_seconds" in summary
        assert "remaining_seconds" in summary
        assert "max_total_seconds" in summary
        assert "within_budget" in summary
        assert summary["max_total_seconds"] == 3.5
        assert isinstance(summary["within_budget"], bool)


# ------------------------------------------------------------------
# generate_with_timeout Tests
# ------------------------------------------------------------------

class TestGenerateWithTimeout:

    @pytest.mark.asyncio
    async def test_completes_within_budget(self):
        """Fast coroutine completes successfully within timeout."""
        async def fast_task():
            await asyncio.sleep(0.01)
            return "success"

        result = await generate_with_timeout(fast_task(), timeout=0.5)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_enforces_timeout_fallback(self):
        """Slow coroutine returns fallback on timeout."""
        async def slow_task():
            await asyncio.sleep(10.0)
            return "should_not_return"

        result = await generate_with_timeout(
            slow_task(),
            timeout=0.05,
            fallback_value="fallback_result",
        )
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_timeout_with_async_function(self):
        """Timeout works with async function (not just coroutine)."""
        async def slow_async_func():
            await asyncio.sleep(10.0)
            return "nope"

        result = await generate_with_timeout(
            slow_async_func,
            timeout=0.05,
            fallback_value="timed_out",
        )
        assert result == "timed_out"

    @pytest.mark.asyncio
    async def test_timeout_with_sync_callable(self):
        """Timeout works with synchronous callable via thread pool."""
        def slow_sync_func():
            time.sleep(10.0)
            return "nope"

        result = await generate_with_timeout(
            slow_sync_func,
            timeout=0.05,
            fallback_value="sync_timed_out",
        )
        assert result == "sync_timed_out"

    @pytest.mark.asyncio
    async def test_exception_returns_fallback(self):
        """Exception in coroutine returns fallback, not raises."""
        async def failing_task():
            raise ValueError("intentional failure")

        result = await generate_with_timeout(
            failing_task(),
            timeout=1.0,
            fallback_value="error_fallback",
        )
        assert result == "error_fallback"

    @pytest.mark.asyncio
    async def test_default_fallback_is_none(self):
        """Default fallback_value is None when not specified."""
        async def slow_task():
            await asyncio.sleep(10.0)
            return "nope"

        result = await generate_with_timeout(slow_task(), timeout=0.05)
        assert result is None


# ------------------------------------------------------------------
# ResponseCache Tests
# ------------------------------------------------------------------

class TestResponseCache:

    def test_cache_hit_performance_under_5ms(self, response_cache):
        """Cache hit must complete in < 5ms."""
        # Pre-populate cache
        response_cache.set(
            message="What services does Hoku offer?",
            intent="general",
            last_3_messages=[],
            response="Hoku offers home healthcare in Pakistan, UAE, and UK.",
            is_emergency=False,
        )

        start = time.perf_counter()
        result = response_cache.get(
            message="What services does Hoku offer?",
            intent="general",
            last_3_messages=[],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert result is not None
        assert "Hoku offers" in result
        assert elapsed_ms < 5.0, f"Cache hit took {elapsed_ms:.3f}ms (max 5ms)"

    def test_cache_miss_returns_none(self, response_cache):
        """Cache miss returns None."""
        result = response_cache.get(
            message="Never seen before query",
            intent="general",
            last_3_messages=[],
        )
        assert result is None

    def test_cache_expiration(self, response_cache):
        """Expired entries are not returned."""
        response_cache.set(
            message="test message",
            intent="general",
            last_3_messages=[],
            response="test response",
            is_emergency=False,
            ttl=0,  # Immediate expiration
        )

        # Small sleep to ensure expiration
        time.sleep(0.01)

        result = response_cache.get(
            message="test message",
            intent="general",
            last_3_messages=[],
        )
        assert result is None

    def test_should_cache_denies_emergency(self, response_cache):
        """Emergency queries must never be cached."""
        assert response_cache.should_cache("general", is_emergency=True) is False
        assert response_cache.should_cache("emergency", is_emergency=False) is False

    def test_should_cache_denies_symptom(self, response_cache):
        """Symptom inquiries must never be cached."""
        assert response_cache.should_cache("symptom", is_emergency=False) is False
        assert response_cache.should_cache("symptom_inquiry", is_emergency=False) is False

    def test_should_cache_allows_general(self, response_cache):
        """General queries may be cached."""
        assert response_cache.should_cache("general", is_emergency=False) is True

    def test_should_cache_allows_booking(self, response_cache):
        """Booking queries may be cached."""
        assert response_cache.should_cache("booking", is_emergency=False) is True

    def test_should_cache_allows_medication(self, response_cache):
        """Medication queries may be cached."""
        assert response_cache.should_cache("medication", is_emergency=False) is True

    def test_should_cache_defaults_deny_unknown(self, response_cache):
        """Unknown intents default to deny (safety-first)."""
        assert response_cache.should_cache("unknown_intent", is_emergency=False) is False

    def test_cache_safety_never_stores_emergency(self, response_cache):
        """set() must refuse to store emergency responses."""
        response_cache.set(
            message="chest pain",
            intent="emergency",
            last_3_messages=[],
            response="EMERGENCY RESPONSE",
            is_emergency=True,
        )

        result = response_cache.get(
            message="chest pain",
            intent="emergency",
            last_3_messages=[],
        )
        assert result is None

    def test_cache_key_determinism(self, response_cache):
        """Same inputs must produce the same key."""
        key1 = response_cache._generate_key(
            "Hello", "general", [{"role": "user", "content": "Hi"}]
        )
        key2 = response_cache._generate_key(
            "Hello", "general", [{"role": "user", "content": "Hi"}]
        )
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex digest length

    def test_cache_key_case_insensitive(self, response_cache):
        """Keys are case-insensitive for message and intent."""
        key1 = response_cache._generate_key("HELLO", "GENERAL", [])
        key2 = response_cache._generate_key("hello", "general", [])
        assert key1 == key2

    def test_cache_clear(self, response_cache):
        """clear() removes all entries and returns count."""
        response_cache.set("msg1", "general", [], "resp1", is_emergency=False)
        response_cache.set("msg2", "booking", [], "resp2", is_emergency=False)
        count = response_cache.clear()
        assert count == 2
        assert response_cache.stats()["entries"] == 0

    def test_cache_stats(self, response_cache):
        """stats() returns entry count and TTL info."""
        response_cache.set("msg", "general", [], "resp", is_emergency=False)
        stats = response_cache.stats()
        assert stats["entries"] == 1
        assert stats["default_ttl_seconds"] == 3600

    def test_cache_invalidate(self, response_cache):
        """invalidate() removes a specific entry."""
        response_cache.set("msg", "general", [], "resp", is_emergency=False)
        removed = response_cache.invalidate("msg", "general", [])
        assert removed is True
        assert response_cache.get("msg", "general", []) is None


# ------------------------------------------------------------------
# LLMFactory Tests
# ------------------------------------------------------------------

class TestLLMFactory:

    def test_compress_prompt_trims_history(self):
        """compress_prompt keeps only last N messages."""
        history = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Reply 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Reply 2"},
            {"role": "user", "content": "Message 3"},
            {"role": "assistant", "content": "Reply 3"},
        ]

        compressed = LLMFactory.compress_prompt(history, max_messages=3)
        assert len(compressed) == 3
        # Should keep the most recent 3
        assert compressed[0]["content"] == "Message 2"

    def test_compress_prompt_truncates_long_messages(self):
        """compress_prompt truncates messages over max_chars_per_turn."""
        long_content = "A" * 1000
        history = [{"role": "user", "content": long_content}]

        compressed = LLMFactory.compress_prompt(
            history, max_messages=3, max_chars_per_turn=600
        )
        assert len(compressed[0]["content"]) == 603  # 600 + "..."
        assert compressed[0]["content"].endswith("...")

    def test_compress_prompt_empty_history(self):
        """compress_prompt handles empty history gracefully."""
        result = LLMFactory.compress_prompt([])
        assert result == []

    def test_compress_prompt_preserves_structure(self):
        """compress_prompt preserves dict structure beyond content."""
        history = [
            {"role": "user", "content": "Hi", "extra": "data"},
            {"role": "assistant", "content": "Hello", "extra": "data"},
        ]
        compressed = LLMFactory.compress_prompt(history, max_messages=1)
        assert len(compressed) == 1
        assert compressed[0]["role"] == "assistant"
        assert compressed[0]["extra"] == "data"

    @patch("app.ai.llm_optimizer.ChatGroq")
    @patch("app.ai.llm_optimizer.ai_settings")
    def test_get_fast_llm_configuration(self, mock_ai_settings, mock_chatgroq):
        """Fast LLM uses correct model and conservative token limit."""
        mock_ai_settings.groq_api_key = "test-key"
        mock_ai_settings.GROQ_FAST_MODEL = "llama-3.1-8b-instant"
        # Day 8.1: request_timeout is derived from INTENT_CLASSIFICATION_TIMEOUT
        mock_ai_settings.INTENT_CLASSIFICATION_TIMEOUT = 1.0

        llm = LLMFactory.get_fast_llm()
        assert llm is not None
        mock_chatgroq.assert_called_once()
        call_kwargs = mock_chatgroq.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.1-8b-instant"
        assert call_kwargs["max_tokens"] == 150
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["request_timeout"] == 1.0

    @patch("app.ai.llm_optimizer.ChatGroq")
    @patch("app.ai.llm_optimizer.ai_settings")
    def test_get_main_llm_configuration(self, mock_ai_settings, mock_chatgroq):
        """Main LLM uses correct model and full token limit from settings."""
        mock_ai_settings.groq_api_key = "test-key"
        mock_ai_settings.GROQ_MAIN_MODEL = "llama-3.3-70b-versatile"
        mock_ai_settings.TEMPERATURE = 0.3
        # Day 8.1: max_tokens and request_timeout are derived from ai_settings
        mock_ai_settings.MAX_TOKENS = 512
        mock_ai_settings.GROQ_TIMEOUT_SECONDS = 3.5

        llm = LLMFactory.get_main_llm()
        assert llm is not None
        mock_chatgroq.assert_called_once()
        call_kwargs = mock_chatgroq.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"
        assert call_kwargs["max_tokens"] == 512
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["request_timeout"] == 3.5

    @patch("app.ai.llm_optimizer.LANGCHAIN_AVAILABLE", False)
    def test_get_fast_llm_unavailable_when_langchain_missing(self):
        """Fast LLM returns None when LangChain is not installed."""
        llm = LLMFactory.get_fast_llm()
        assert llm is None

    @patch("app.ai.llm_optimizer.LANGCHAIN_AVAILABLE", False)
    def test_get_main_llm_unavailable_when_langchain_missing(self):
        """Main LLM returns None when LangChain is not installed."""
        llm = LLMFactory.get_main_llm()
        assert llm is None


# ------------------------------------------------------------------
# Fallback Responses Tests
# ------------------------------------------------------------------

class TestFallbackResponses:

    def test_fallback_general_contains_disclaimer(self):
        """All fallback responses must include the clinical disclaimer."""
        from app.utils.constants import SAFETY_DISCLAIMER
        assert SAFETY_DISCLAIMER in FALLBACK_GENERAL

    def test_fallback_emergency_contains_disclaimer(self):
        assert "Please consult a doctor for proper diagnosis." in FALLBACK_EMERGENCY

    def test_fallback_booking_contains_disclaimer(self):
        assert "Please consult a doctor for proper diagnosis." in FALLBACK_BOOKING

    def test_get_fallback_for_intent_emergency(self):
        """Emergency intent returns emergency fallback."""
        result = get_fallback_for_intent("emergency")
        assert result == FALLBACK_EMERGENCY

    def test_get_fallback_for_intent_booking(self):
        """Booking intent returns booking fallback."""
        result = get_fallback_for_intent("booking")
        assert result == FALLBACK_BOOKING

    def test_get_fallback_for_intent_general(self):
        """General intent returns general fallback."""
        result = get_fallback_for_intent("general")
        assert result == FALLBACK_GENERAL

    def test_get_fallback_for_intent_unknown(self):
        """Unknown intent defaults to general fallback."""
        result = get_fallback_for_intent("unknown")
        assert result == FALLBACK_GENERAL

    def test_get_fallback_case_insensitive(self):
        """Intent matching is case-insensitive."""
        assert get_fallback_for_intent("EMERGENCY") == FALLBACK_EMERGENCY
        assert get_fallback_for_intent("Booking") == FALLBACK_BOOKING


# ------------------------------------------------------------------
# Integration Tests — Budget + Timeout + Cache
# ------------------------------------------------------------------

class TestPerformanceIntegration:

    @pytest.mark.asyncio
    async def test_full_pipeline_budget_enforcement(self, optimizer):
        """Simulate a full pipeline with budget checks at each stage."""
        overall_start = time.perf_counter()

        # Step 1: Emergency detection (simulated fast)
        step_start = time.perf_counter()
        time.sleep(0.001)  # Simulate 1ms work
        assert optimizer.enforce_budget("emergency_detect", step_start) is True

        # Step 2: Intent classification (simulated)
        step_start = time.perf_counter()
        time.sleep(0.005)  # Simulate 5ms work
        assert optimizer.enforce_budget("intent_classify", step_start) is True

        # Step 3: Check remaining budget for LLM
        remaining = optimizer.remaining_budget(overall_start)
        assert remaining > 0  # Should still have budget left

        # Simulate LLM with timeout
        async def mock_llm():
            await asyncio.sleep(0.01)
            return "Mock response"

        result = await generate_with_timeout(mock_llm(), timeout=remaining)
        assert result == "Mock response"

    @pytest.mark.asyncio
    async def test_cache_short_circuits_pipeline(self, response_cache):
        """Cache hit should return instantly, skipping expensive stages."""
        # Pre-populate cache
        response_cache.set(
            message="What are your services?",
            intent="general",
            last_3_messages=[],
            response="We offer home healthcare in Pakistan, UAE, and UK.",
            is_emergency=False,
        )

        start = time.perf_counter()
        cached = response_cache.get(
            message="What are your services?",
            intent="general",
            last_3_messages=[],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert cached is not None
        assert elapsed_ms < 5.0  # Must be under 5ms

    def test_budget_summary_after_overrun(self, optimizer):
        """Budget summary correctly reports when total time is exceeded."""
        start = time.perf_counter() - 5.0  # Simulate 5s elapsed
        summary = optimizer.get_budget_summary(start)
        assert summary["within_budget"] is False
        assert summary["remaining_seconds"] < 0