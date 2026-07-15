"""
Hoku Health Care - Chatbot Unit Tests (Day 3).

Tests for HokuChatbot with LLMChain and conversation memory.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from langchain.memory import ConversationBufferMemory

from app.ai.chatbot import HokuChatbot
from app.ai.utils import (
    extract_json_from_response,
    parse_severity_from_response,
    parse_should_see_doctor,
    parse_specialist_from_response,
)
from app.utils.constants import SAFETY_DISCLAIMER


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.groq_api_key = "test-api-key"
    settings.GROQ_MAIN_MODEL = "llama-3.3-70b-versatile"
    settings.GROQ_FAST_MODEL = "llama-3.1-8b-instant"
    settings.TEMPERATURE = 0.3
    settings.MAX_TOKENS = 512
    settings.GROQ_TIMEOUT_SECONDS = 3.5
    settings.MAX_RETRIES = 3
    settings.MEMORY_MESSAGE_LIMIT = 10
    settings.MEMORY_MAX_TOKENS = 307
    return settings


@pytest.fixture
def chatbot(mock_settings):
    with patch("app.ai.chatbot.ai_settings", mock_settings):
        bot = HokuChatbot()
        yield bot


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_memory():
    memory = ConversationBufferMemory(memory_key="history", input_key="message")
    return memory


@pytest.fixture
def mock_llm_chain():
    """Mock LLMChain returning dict with 'text' key."""
    chain = MagicMock()
    chain.invoke.return_value = {
        "text": json.dumps({
            "reply": "You may be experiencing mild dehydration. Drink water. " + SAFETY_DISCLAIMER,
            "suggestedSpecialist": "General Physician",
            "severity": "mild",
            "shouldSeeDoctor": False,
        }),
    }
    return chain


class TestHokuChatbotInitialization:
    def test_init_sets_config(self, chatbot, mock_settings):
        assert chatbot.main_model == "llama-3.3-70b-versatile"
        assert chatbot.fast_model == "llama-3.1-8b-instant"
        assert chatbot.temperature == 0.3

    def test_lazy_llm_loading(self, chatbot):
        assert chatbot._main_llm is None
        assert chatbot._fast_llm is None

    def test_fallback_when_no_api_key(self, mock_settings):
        mock_settings.groq_api_key = ""
        with patch("app.ai.chatbot.ai_settings", mock_settings):
            bot = HokuChatbot()
            assert bot.main_llm is None


class TestGetResponse:
    async def test_returns_correct_keys(self, chatbot, mock_llm_chain, mock_db, mock_memory):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert "reply" in result
                assert "suggestedSpecialist" in result
                assert "severity" in result
                assert "shouldSeeDoctor" in result

    async def test_reply_contains_safety_disclaimer(self, chatbot, mock_llm_chain, mock_db, mock_memory):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert SAFETY_DISCLAIMER in result["reply"]

    async def test_parses_specialist_correctly(self, chatbot, mock_llm_chain, mock_db, mock_memory):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert result["suggestedSpecialist"] == "General Physician"

    async def test_parses_severity_correctly(self, chatbot, mock_llm_chain, mock_db, mock_memory):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert result["severity"] == "mild"

    async def test_parses_should_see_doctor_correctly(self, chatbot, mock_llm_chain, mock_db, mock_memory):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert result["shouldSeeDoctor"] is False

    async def test_fallback_on_timeout(self, chatbot, mock_db, mock_memory):
        slow_chain = MagicMock()
        slow_chain.invoke = lambda **kwargs: time.sleep(10) or {}
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=slow_chain):
                chatbot.timeout = 0.01
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert "sorry" in result["reply"].lower()
                assert SAFETY_DISCLAIMER in result["reply"]
                assert result["shouldSeeDoctor"] is True

    async def test_fallback_on_llm_error(self, chatbot, mock_db, mock_memory):
        bad_chain = MagicMock()
        bad_chain.invoke.side_effect = Exception("Groq API error")
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=bad_chain):
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert "sorry" in result["reply"].lower()
                assert SAFETY_DISCLAIMER in result["reply"]
                assert result["shouldSeeDoctor"] is True

    async def test_response_time_under_nfr(self, chatbot, mock_llm_chain, mock_db, mock_memory):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                start = time.perf_counter()
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                elapsed = time.perf_counter() - start
                assert elapsed < 4.0

    async def test_fallback_when_llm_none(self, chatbot, mock_db):
        chatbot._main_llm = None
        result = await chatbot.get_response("test", user_id=1, db=mock_db)
        assert "sorry" in result["reply"].lower()
        assert result["shouldSeeDoctor"] is True


class TestParseLlmOutput:
    def test_direct_json_parse(self, chatbot):
        text = json.dumps({
            "reply": "Test reply " + SAFETY_DISCLAIMER,
            "suggestedSpecialist": "Dermatologist",
            "severity": "moderate",
            "shouldSeeDoctor": True,
        })
        result = chatbot._parse_llm_output(text)
        assert result["reply"] == "Test reply " + SAFETY_DISCLAIMER
        assert result["suggestedSpecialist"] == "Dermatologist"

    def test_markdown_json_extraction(self, chatbot):
        text = '```json\n{"reply": "Hello ' + SAFETY_DISCLAIMER + '", "severity": "mild"}\n```'
        result = chatbot._parse_llm_output(text)
        assert "Hello" in result["reply"]

    def test_regex_fallback(self, chatbot):
        text = 'Some text "severity": "severe" more text "shouldSeeDoctor": true'
        result = chatbot._parse_llm_output(text)
        assert result["severity"] == "severe"
        assert result["shouldSeeDoctor"] is True

    def test_plain_text_fallback(self, chatbot):
        text = "This is just plain text with no JSON."
        result = chatbot._parse_llm_output(text)
        assert result["reply"] == text
        assert result["severity"] == "unknown"

    def test_extract_text_from_dict(self, chatbot):
        assert chatbot._extract_text_from_result({"text": "hello"}) == "hello"

    def test_extract_text_from_string(self, chatbot):
        assert chatbot._extract_text_from_result("hello") == "hello"


class TestUtilityFunctions:
    def test_parse_specialist_from_json(self):
        text = '{"suggestedSpecialist": "Cardiologist"}'
        assert parse_specialist_from_response(text) == "Cardiologist"

    def test_parse_specialist_null_returns_none(self):
        text = '{"suggestedSpecialist": null}'
        assert parse_specialist_from_response(text) is None

    def test_parse_specialist_regex_fallback(self):
        text = '"suggestedSpecialist": "Neurologist"'
        assert parse_specialist_from_response(text) == "Neurologist"

    def test_parse_severity_valid(self):
        text = '{"severity": "severe"}'
        assert parse_severity_from_response(text) == "severe"

    def test_parse_severity_invalid_defaults_unknown(self):
        text = '{"severity": "critical"}'
        assert parse_severity_from_response(text) == "unknown"

    def test_parse_should_see_doctor_true(self):
        text = '{"shouldSeeDoctor": true}'
        assert parse_should_see_doctor(text) is True

    def test_parse_should_see_doctor_safety_bias(self):
        text = "some random text without any flag"
        assert parse_should_see_doctor(text) is True

    def test_extract_json_from_markdown(self):
        text = '```json\n{"key": "value"}\n```'
        assert extract_json_from_response(text) == {"key": "value"}

    def test_extract_json_invalid_returns_empty(self):
        text = "not json at all"
        assert extract_json_from_response(text) == {}