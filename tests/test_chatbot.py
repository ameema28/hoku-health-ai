"""
Hoku Health Care - Chatbot Unit Tests (Day 3, hardened Day 8.1).

Tests for HokuChatbot with LLMChain and conversation memory.

Day 8.1 hardening
-----------------
The original suite patched LLMChain with a bare MagicMock. A bare MagicMock
accepts ANY call signature silently, so `chain.invoke(invoke_input={...})`
passed in tests while raising

    TypeError: Chain.invoke() missing 1 required positional argument: 'input'

against the real LangChain class on every live request. 215 tests were green
while the endpoint was 100% broken for non-emergency traffic.

The fix is `_make_chain_mock()`, which builds a mock whose `invoke` carries the
REAL LangChain signature via `create_autospec`. Any future signature mismatch
now fails at test time, not on Swagger.

Rule for this module: never hand-roll `MagicMock()` for a LangChain object.
"""

import inspect
import json
import time
from unittest.mock import MagicMock, PropertyMock, create_autospec, patch

import pytest
from langchain.memory import ConversationBufferMemory

try:
    from langchain.chains import LLMChain
except ImportError:  # pragma: no cover - minimal CI image
    LLMChain = None

from app.ai.chatbot import HokuChatbot
from app.ai.intent_classifier import IntentEnum
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


def _chain_invoke_spec(input, config=None, **kwargs):  # noqa: A002 - mirrors LangChain
    """
    Signature mirror of langchain.chains.base.Chain.invoke.

    Used as the autospec template so the payload MUST be supplied as the
    parameter literally named `input` (positionally or by that keyword).
    Passing `invoke_input=...` — the Day 8 production bug — raises TypeError
    here exactly as it does against the real class.
    """
    raise NotImplementedError  # pragma: no cover - never executed


def _make_chain_mock(return_value):
    """
    Build an LLMChain test double whose `invoke` enforces the REAL signature.

    Always use this instead of a bare MagicMock. A bare MagicMock accepts any
    keyword name silently, which is how the Day 8 TypeError shipped behind a
    fully green suite.
    """
    chain = MagicMock()
    chain.invoke = create_autospec(_chain_invoke_spec, return_value=return_value)
    return chain


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_memory():
    memory = ConversationBufferMemory(memory_key="history", input_key="message")
    return memory


@pytest.fixture
def mock_llm_chain():
    """Signature-enforcing LLMChain mock returning a dict with a 'text' key."""
    return _make_chain_mock({
        "text": json.dumps({
            "reply": "You may be experiencing mild dehydration. Drink water. " + SAFETY_DISCLAIMER,
            "suggestedSpecialist": "General Physician",
            "severity": "mild",
            "shouldSeeDoctor": False,
        }),
    })


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
        # Day 8.1: the old stub was `lambda **kwargs: ...`, which could not
        # accept a positional payload at all. It "passed" by raising TypeError
        # and landing in the fallback branch — testing nothing about timeouts.
        # This version accepts the real signature and actually blocks.
        slow_chain = MagicMock()

        def _slow_invoke(input, config=None, **kwargs):  # noqa: A002
            time.sleep(10)
            return {}

        slow_chain.invoke = _slow_invoke
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=slow_chain):
                chatbot.timeout = 0.01
                result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                assert "sorry" in result["reply"].lower()
                assert SAFETY_DISCLAIMER in result["reply"]
                assert result["shouldSeeDoctor"] is True

    async def test_fallback_on_llm_error(self, chatbot, mock_db, mock_memory):
        bad_chain = _make_chain_mock(None)
        # A transport-style failure must still degrade to the safe fallback.
        bad_chain.invoke.side_effect = ConnectionError("Groq API error")
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
        # Day 8.1: `chatbot._main_llm = None` does NOT disable the LLM —
        # main_llm is a lazy property, so the very next access re-initialises
        # it. The test previously "passed" only because constructing ChatGroq
        # with a fake key still produced an object, whose call then failed over
        # the network into the fallback branch. That is a live HTTP request
        # inside a unit test, and it silently stopped working the moment LLM
        # construction was routed through LLMFactory.
        #
        # Patch the property itself so the unavailable-LLM path is exercised
        # deterministically and offline.
        with patch.object(
            type(chatbot), "main_llm", new_callable=PropertyMock, return_value=None
        ):
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


class TestIntentFields:
    @pytest.mark.asyncio
    async def test_response_includes_intent(self, chatbot, mock_llm_chain, mock_db, mock_memory):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                with patch.object(chatbot.intent_classifier, "classify_intent", return_value=(IntentEnum.SYMPTOM, 0.92)):
                    result = await chatbot.get_response("I have a headache", user_id=1, db=mock_db)
                    assert "intent" in result
                    assert "confidence" in result
                    assert result["intent"] == "symptom"
                    assert result["confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_emergency_bypass_returns_emergency_intent(self, chatbot, mock_db):
        result = await chatbot.get_response("I can't breathe, chest pain", user_id=1, db=mock_db)
        assert result["intent"] == "emergency"
        assert result["confidence"] == 1.0
        assert "🚨" in result["reply"]

    @pytest.mark.asyncio
    async def test_fallback_includes_intent_fields(self, chatbot, mock_db):
        # Day 8.1: patch the property, not the backing field — see
        # test_fallback_when_llm_none for why.
        with patch.object(
            type(chatbot), "main_llm", new_callable=PropertyMock, return_value=None
        ):
            with patch.object(chatbot.intent_classifier, "classify_intent", return_value=(IntentEnum.GENERAL, 0.0)):
                result = await chatbot.get_response("test", user_id=1, db=mock_db)
                assert "intent" in result
                assert "confidence" in result

class TestChainInvokeContract:
    """
    Day 8.1 regression guard for the live-only TypeError.

    These tests fail against the OLD chatbot.py (which passed the payload as
    `invoke_input=`) and pass against the fixed version. Keep them.
    """

    @pytest.mark.asyncio
    async def test_invoke_receives_payload_positionally(
        self, chatbot, mock_llm_chain, mock_db, mock_memory
    ):
        with patch("app.ai.chatbot.HokuConversationMemory") as MockMem:
            MockMem.return_value.load_memory.return_value = mock_memory
            with patch("app.ai.chatbot.LLMChain", return_value=mock_llm_chain):
                await chatbot.get_response("I have a headache", user_id=1, db=mock_db)

        assert mock_llm_chain.invoke.called, "chain.invoke was never reached"
        call_args, call_kwargs = mock_llm_chain.invoke.call_args

        # The payload must arrive as the first positional argument.
        assert call_args, "payload was not passed positionally"
        payload = call_args[0]
        assert isinstance(payload, dict)
        assert "message" in payload and "context" in payload

        # The exact keyword that broke production must never reappear.
        assert "invoke_input" not in call_kwargs

    @pytest.mark.asyncio
    async def test_signature_mock_rejects_wrong_keyword(self, mock_llm_chain):
        """Proof the mock is strict enough to catch the original bug."""
        with pytest.raises(TypeError):
            mock_llm_chain.invoke(invoke_input={"message": "hi", "context": ""})


class TestSafetyConvergence:
    """
    Day 8.1 guard: every validate pattern must have a converging sanitiser.

    A `_DIAGNOSIS_PATTERNS` / `_PRESCRIPTION_PATTERNS` entry with no matching
    replacement in `sanitize_response` — or a replacement that reproduces its
    own trigger phrase — cannot converge. The 3-strike loop then burns all
    attempts and returns SAFETY_FALLBACK_RESPONSE, silently destroying a
    correct and safe answer. Two such defects shipped in Day 7.
    """

    BENIGN = [
        "If you have any questions, please reach out to our team.",
        "Since you have been experiencing fatigue, rest and hydration help.",
        "You have several options for booking an appointment.",
        "You may want to rest and drink fluids.",
    ]

    UNSAFE = [
        "You have diabetes and need insulin.",
        "Take 500 mg paracetamol twice daily.",
        "Your symptoms indicate a serious infection.",
        "You are suffering from bronchitis.",
        "It is clear that you have pneumonia.",
        "This confirms a viral infection.",
        "Stop taking metformin immediately.",
    ]

    @pytest.mark.parametrize("text", BENIGN)
    def test_benign_text_is_not_flagged(self, text, mock_db):
        """Ordinary phrasing must never be mistaken for a diagnosis."""
        from app.ai.safety_guardrails import SafetyGuardrails

        result, violations, severity = SafetyGuardrails.apply_3_strike_safety(
            text=f"{text} {SAFETY_DISCLAIMER}", user_id=1, db=None
        )
        assert "unable to provide a medical opinion" not in result, (
            f"3-strike fallback fired on benign text: {text!r}"
        )
        assert text.split(",")[0][:20] in result

    @pytest.mark.parametrize("text", UNSAFE)
    def test_unsafe_text_converges_without_fallback(self, text):
        """Unsafe text must be sanitised into a safe form, not fallback-ed."""
        from app.ai.safety_guardrails import SafetyGuardrails

        result, violations, severity = SafetyGuardrails.apply_3_strike_safety(
            text=f"{text} {SAFETY_DISCLAIMER}", user_id=1, db=None
        )
        assert violations, "expected at least one violation to be recorded"
        assert "unable to provide a medical opinion" not in result, (
            f"sanitiser failed to converge for {text!r} — check that its "
            f"replacement does not reproduce its own trigger phrase"
        )
        assert SAFETY_DISCLAIMER in result

    def test_no_replacement_reproduces_its_own_trigger(self):
        """
        Structural guard: re-running the sanitiser must reach a fixed point.

        If sanitize_response(sanitize_response(x)) != sanitize_response(x),
        some replacement re-matches its own output.
        """
        from app.ai.safety_guardrails import SafetyGuardrails

        for text in self.UNSAFE:
            once = SafetyGuardrails.sanitize_response(f"{text} {SAFETY_DISCLAIMER}")
            twice = SafetyGuardrails.sanitize_response(once)
            assert once == twice, f"sanitiser is not idempotent for {text!r}"


class TestResponseCacheKeys:
    """Day 8.1 guard: the cache must actually be able to hit."""

    def test_general_intent_key_ignores_history(self):
        """
        A repeated general question must produce the SAME key even though
        conversation history has grown in between — otherwise the cache can
        never register a hit in production.
        """
        from app.ai.caching import ResponseCache

        cache = ResponseCache()
        question = "What services does Hoku Health Care offer?"

        key_turn_1 = cache._generate_key(question, "general", [])
        key_turn_2 = cache._generate_key(
            question, "general", ["user asked before", "assistant replied before"]
        )
        assert key_turn_1 == key_turn_2

    def test_non_general_intent_stays_context_sensitive(self):
        from app.ai.caching import ResponseCache

        cache = ResponseCache()
        question = "What about the dosage?"

        key_a = cache._generate_key(question, "medication", ["about paracetamol"])
        key_b = cache._generate_key(question, "medication", ["about metformin"])
        assert key_a != key_b, "context must still disambiguate follow-up questions"

    @pytest.mark.parametrize("intent", ["symptom", "emergency"])
    def test_clinical_intents_are_never_cached(self, intent):
        from app.ai.caching import ResponseCache

        assert ResponseCache().should_cache(intent, is_emergency=False) is False


class TestCompressedHistoryApplication:
    """Day 8.1 guard: compress_prompt returns dicts, memory stores objects."""

    def test_dict_history_is_converted_and_applied(self):
        from langchain_core.messages import AIMessage, HumanMessage

        from app.ai.chatbot import HokuChatbot

        memory = ConversationBufferMemory(memory_key="history", input_key="message")
        memory.chat_memory.add_user_message("old one")
        memory.chat_memory.add_ai_message("old two")
        memory.chat_memory.add_user_message("old three")

        compressed = [
            {"role": "user", "content": "kept question"},
            {"role": "assistant", "content": "kept answer"},
        ]
        HokuChatbot._apply_compressed_history(memory, compressed)

        messages = memory.chat_memory.messages
        assert len(messages) == 2, "compression was not applied to the buffer"
        assert isinstance(messages[0], HumanMessage)
        assert isinstance(messages[1], AIMessage)
        assert messages[0].content == "kept question"

    def test_unrecognised_shape_leaves_buffer_untouched(self):
        from app.ai.chatbot import HokuChatbot

        memory = ConversationBufferMemory(memory_key="history", input_key="message")
        memory.chat_memory.add_user_message("must survive")

        HokuChatbot._apply_compressed_history(memory, [{"role": "martian", "content": "x"}])
        assert len(memory.chat_memory.messages) == 1
        assert memory.chat_memory.messages[0].content == "must survive"


class TestPerformanceBudgets:
    """Day 8.1 guard: concurrent stages must not be summed."""

    def test_serial_path_fits_inside_ceiling(self):
        from app.ai.ai_performance import ResponseOptimizer

        assert ResponseOptimizer.expected_serial_time() <= ResponseOptimizer.MAX_TOTAL_TIME

    def test_intent_budget_matches_its_timeout(self):
        from app.ai.ai_performance import ResponseOptimizer
        from app.ai.config import ai_settings

        # The budget is DERIVED from the setting, so this holds for any .env
        # value. A hardcoded budget would drift the moment the timeout moved —
        # which is exactly what happened when .env set the timeout to 1.5.
        assert (
            ResponseOptimizer.TIME_BUDGETS["intent_classify"]
            >= ai_settings.INTENT_CLASSIFICATION_TIMEOUT
        ), "budget below its own timeout guarantees spurious warnings"


class TestProgrammingErrorsSurface:
    """Day 8.1 guard: signature bugs must not masquerade as timeouts."""

    @pytest.mark.asyncio
    async def test_type_error_is_reraised_not_swallowed(self):
        from app.ai.ai_performance import generate_with_timeout

        def needs_an_argument(required):
            return required

        with pytest.raises(TypeError):
            await generate_with_timeout(
                coro_or_func=needs_an_argument, timeout=1.0, fallback_value="MASKED"
            )

    @pytest.mark.asyncio
    async def test_transport_error_still_degrades(self):
        from app.ai.ai_performance import generate_with_timeout

        def flaky():
            raise ConnectionError("groq unreachable")

        result = await generate_with_timeout(
            coro_or_func=flaky, timeout=1.0, fallback_value="FALLBACK"
        )
        assert result == "FALLBACK"