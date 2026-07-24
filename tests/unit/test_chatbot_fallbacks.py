"""
Hoku Health Care - Day 10 chatbot fallback-path coverage.

Targets the error and degradation branches of HokuChatbot.get_response
that the happy-path integration tests never reach: main-LLM-unavailable,
LLM timeout, and malformed-output parsing. All Groq interaction is
mocked; these are fast, deterministic unit tests that also assert the
clinical safety contract (disclaimer always present) on every path.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.ai.chatbot import HokuChatbot
from app.utils.constants import SAFETY_DISCLAIMER


@pytest.fixture
def mock_db() -> MagicMock:
    """A stand-in DB session; the chatbot only passes it through."""
    return MagicMock()


@pytest.fixture
def mock_memory() -> MagicMock:
    """Empty conversation memory so no history is loaded."""
    from langchain.memory import ConversationBufferMemory

    return ConversationBufferMemory(memory_key="history", input_key="message")


class TestChatbotFallbackPaths:
    """Degradation branches must still return a safe, disclaimer-bearing reply."""

    @pytest.mark.asyncio
    async def test_main_llm_unavailable_returns_fallback(
        self, mock_db: MagicMock, mock_memory: MagicMock
    ) -> None:
        """When the main LLM can't initialize, a safe fallback is returned."""
        bot = HokuChatbot()
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem, patch.object(
            type(bot), "main_llm", new_callable=lambda: property(lambda self: None)
        ):
            MockMem.return_value.load_memory.return_value = mock_memory
            result = await bot.get_response("Tell me about wellness", user_id=1, db=mock_db)

        assert "reply" in result
        assert SAFETY_DISCLAIMER in result["reply"]
        assert result["shouldSeeDoctor"] is True
        assert result["intent"] in (
            "general", "symptom", "booking", "medication", "emergency",
        )

    @pytest.mark.asyncio
    async def test_llm_timeout_returns_fallback(
        self, mock_db: MagicMock, mock_memory: MagicMock
    ) -> None:
        """A None result from the timeout wrapper yields the safe fallback."""
        bot = HokuChatbot()
        chain = MagicMock()
        chain.invoke = MagicMock(return_value=None)

        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem, patch(
            "app.ai.chatbot.LLMChain", return_value=chain
        ), patch(
            "app.ai.chatbot.generate_with_timeout", return_value=None
        ):
            MockMem.return_value.load_memory.return_value = mock_memory
            result = await bot.get_response("General question", user_id=1, db=mock_db)

        assert SAFETY_DISCLAIMER in result["reply"]
        assert result["shouldSeeDoctor"] is True

    @pytest.mark.asyncio
    async def test_malformed_llm_output_still_safe(
        self, mock_db: MagicMock, mock_memory: MagicMock
    ) -> None:
        """Non-JSON LLM text is parsed defensively and gets the disclaimer."""
        bot = HokuChatbot()
        chain = MagicMock()
        chain.invoke = MagicMock(return_value={"text": "just some plain prose, no json here"})

        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem, patch(
            "app.ai.chatbot.LLMChain", return_value=chain
        ):
            MockMem.return_value.load_memory.return_value = mock_memory
            result = await bot.get_response("Any tips?", user_id=1, db=mock_db)

        assert SAFETY_DISCLAIMER in result["reply"]


class TestParseAndFallbackHelpers:
    """Direct tests of the small pure helpers on HokuChatbot."""

    def test_parse_llm_output_direct_json(self) -> None:
        """A clean JSON payload is parsed into the expected keys."""
        bot = HokuChatbot()
        payload = json.dumps({
            "reply": "Rest well.",
            "suggestedSpecialist": "General Physician",
            "severity": "mild",
            "shouldSeeDoctor": False,
        })
        parsed = bot._parse_llm_output(payload)
        assert parsed["reply"] == "Rest well."
        assert parsed["suggestedSpecialist"] == "General Physician"
        assert parsed["severity"] == "mild"
        assert parsed["shouldSeeDoctor"] is False

    def test_parse_llm_output_plain_text_defaults(self) -> None:
        """Plain text falls back to safe defaults (shouldSeeDoctor True)."""
        bot = HokuChatbot()
        parsed = bot._parse_llm_output("no structure at all")
        assert parsed["shouldSeeDoctor"] is True
        assert parsed["severity"] == "unknown"

    def test_fallback_response_contains_disclaimer(self) -> None:
        """The hardcoded fallback always carries the disclaimer."""
        bot = HokuChatbot()
        fallback = bot._fallback_response("test reason")
        assert SAFETY_DISCLAIMER in fallback["reply"]
        assert fallback["shouldSeeDoctor"] is True

    def test_extract_text_from_string_and_dict(self) -> None:
        """Text extraction handles both str and dict chain returns."""
        bot = HokuChatbot()
        assert bot._extract_text_from_result("hello") == "hello"
        assert bot._extract_text_from_result({"text": "hi"}) == "hi"
