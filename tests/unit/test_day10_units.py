"""
Hoku Health Care - Day 10 unit tests.

Fast, isolated tests for the Day 10 building blocks: the sliding-window
rate limiter, the structured-logging redaction layer, and the input
validators. No Groq, no HTTP, no DB - pure function calls - so these run
in milliseconds and lift coverage on code that the integration tests only
touch incidentally.
"""

from __future__ import annotations

import logging

import pytest

from app.core.logging import (
    JSONFormatter,
    RedactionFilter,
    redact,
    set_correlation_id,
)
from app.core.rate_limit import (
    RateLimitExceeded,
    SlidingWindowRateLimiter,
)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
class TestSlidingWindowRateLimiter:
    """Behavioural tests for the per-user sliding-window limiter."""

    def test_admits_up_to_limit(self) -> None:
        """The first `limit` requests in a window are all admitted."""
        limiter = SlidingWindowRateLimiter(limit=5, window_seconds=60)
        base = 1000.0
        for i in range(5):
            state = limiter.check("user-a", now=base + i * 0.1)
        assert state.limit == 5
        assert state.remaining == 0

    def test_blocks_over_limit_with_retry_after(self) -> None:
        """The (limit+1)-th request raises with a positive retry_after."""
        limiter = SlidingWindowRateLimiter(limit=3, window_seconds=60)
        base = 2000.0
        for i in range(3):
            limiter.check("user-b", now=base + i * 0.1)
        with pytest.raises(RateLimitExceeded) as exc_info:
            limiter.check("user-b", now=base + 0.4)
        assert exc_info.value.limit == 3
        assert exc_info.value.retry_after_seconds >= 1

    def test_users_are_isolated(self) -> None:
        """One user's exhaustion does not affect another user."""
        limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
        base = 3000.0
        limiter.check("user-c", now=base)
        limiter.check("user-c", now=base + 0.1)
        # user-c is now full; user-d must still be admitted.
        state = limiter.check("user-d", now=base + 0.2)
        assert state.remaining == 1

    def test_window_slides(self) -> None:
        """Old timestamps age out, freeing the quota."""
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
        base = 4000.0
        limiter.check("user-e", now=base)
        with pytest.raises(RateLimitExceeded):
            limiter.check("user-e", now=base + 30)
        # After the window fully passes, a new request is admitted.
        state = limiter.check("user-e", now=base + 61)
        assert state.remaining == 0

    def test_reset_clears_state(self) -> None:
        """reset() frees a key immediately."""
        limiter = SlidingWindowRateLimiter(limit=1, window_seconds=60)
        base = 5000.0
        limiter.check("user-f", now=base)
        limiter.reset("user-f")
        # Reset means the next call is treated as the first in the window.
        state = limiter.check("user-f", now=base + 1)
        assert state.remaining == 0

    def test_limit_floor_is_one(self) -> None:
        """A limit below 1 is clamped up to 1."""
        limiter = SlidingWindowRateLimiter(limit=0, window_seconds=60)
        assert limiter.limit == 1


# ---------------------------------------------------------------------------
# Redaction / structured logging
# ---------------------------------------------------------------------------
class TestRedaction:
    """The redaction helpers must strip PHI and secrets."""

    def test_redact_email(self) -> None:
        """Email addresses are masked."""
        assert "patient@example.com" not in redact("contact patient@example.com now")

    def test_redact_api_key(self) -> None:
        """Groq-style API keys are masked."""
        assert "gsk_" not in redact("key is gsk_ABCDEFGHIJKLMNOPQRSTUV")

    def test_redact_jwt(self) -> None:
        """JWT-shaped tokens are masked."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.abcdef123456"
        assert jwt not in redact(f"Authorization: Bearer {jwt}")

    def test_redact_postgres_credentials(self) -> None:
        """Inline DB credentials are masked but the scheme survives."""
        out = redact("postgresql://hoku:secretpw@db:5432/hoku_health")
        assert "secretpw" not in out
        assert out.startswith("postgresql://")

    def test_redact_leaves_clean_text_untouched(self) -> None:
        """Text with no secrets passes through unchanged."""
        clean = "Intent classified as general with confidence 0.88"
        assert redact(clean) == clean


class TestRedactionFilter:
    """The logging.Filter form scrubs records before they are emitted."""

    def _record(self, msg: str, **extra: object) -> logging.LogRecord:
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg=msg, args=(), exc_info=None,
        )
        for key, value in extra.items():
            setattr(record, key, value)
        return record

    def test_forbidden_extra_key_is_redacted(self) -> None:
        """A record carrying user_message has it replaced with [REDACTED]."""
        flt = RedactionFilter()
        record = self._record("turn done", user_message="I have chest pain")
        flt.filter(record)
        assert record.user_message == "[REDACTED]"

    def test_message_secret_is_masked(self) -> None:
        """A secret interpolated into the message text is masked."""
        flt = RedactionFilter()
        record = self._record("emailing patient@example.com")
        flt.filter(record)
        assert "patient@example.com" not in record.getMessage()

    def test_filter_always_returns_true(self) -> None:
        """Records are scrubbed, never dropped."""
        flt = RedactionFilter()
        assert flt.filter(self._record("nothing sensitive")) is True


class TestJSONFormatter:
    """The JSON formatter emits parseable, correlation-tagged lines."""

    def test_format_includes_correlation_id(self) -> None:
        """The bound correlation id appears in the rendered JSON."""
        import json

        set_correlation_id("test-corr-id")
        formatter = JSONFormatter(environment="test")
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname=__file__, lineno=1,
            msg="hello", args=(), exc_info=None,
        )
        payload = json.loads(formatter.format(record))
        assert payload["correlation_id"] == "test-corr-id"
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["environment"] == "test"


# ---------------------------------------------------------------------------
# Validators (pre-existing, thinly covered)
# ---------------------------------------------------------------------------
class TestValidators:
    """Input sanitisation and length validation."""

    def test_sanitize_message_strips_and_escapes(self) -> None:
        """sanitize_message returns a cleaned, non-empty string for normal input."""
        from app.utils.validators import sanitize_message

        out = sanitize_message("  I have a headache  ")
        assert out
        assert "headache" in out

    def test_validate_message_length_accepts_normal(self) -> None:
        """A normal-length message passes validation."""
        from app.utils.validators import validate_message_length

        assert validate_message_length("I have a fever") is True

    def test_validate_message_length_rejects_overlong(self) -> None:
        """A message beyond the max length is rejected."""
        from app.utils.validators import validate_message_length

        assert validate_message_length("x" * 5000) is False
