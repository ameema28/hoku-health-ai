"""
Deterministic fallback replies (built from retrieved FAQ and doctor data),
the fallback_used flag on every reply path, and the reply prompt rules.
All LLM calls are mocked.
"""

import json
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.ai import fallback_responses as fr
from app.ai.chatbot import HokuChatbot
from app.ai.intent_classifier import IntentEnum
from app.ai.prompts import (
    REPLY_RULES,
    RAG_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    chat_prompt_template,
    rag_chat_prompt_template,
)
from app.ai.safety_guardrails import SafetyGuardrails
from app.schemas.schemas_doctor import DoctorSuggestion
from app.scripts.seed_faqs import FAQS
from app.utils.constants import SAFETY_DISCLAIMER

FAQ_ANSWER = FAQS[7]["answer"]  # home-visit wait time


def _doc(answer: str = FAQ_ANSWER) -> Document:
    return Document(
        page_content="Q/A",
        metadata={"question": "q", "answer": answer, "score": 0.8},
    )


def _doctor(requested=None) -> DoctorSuggestion:
    return DoctorSuggestion(
        specialist="Dermatologist",
        doctor_name="Dr. Sana Malik",
        experience=9,
        availability=None,
        doctor_id=3,
        requested_specialist=requested,
    )


# ---------------------------------------------------------------------------
# build_fallback_reply
# ---------------------------------------------------------------------------
class TestBuildFallbackReply:
    def test_faq_answer_then_disclaimer(self):
        reply = fr.build_fallback_reply("general", [_doc()], None)
        assert reply == f"{FAQ_ANSWER} {SAFETY_DISCLAIMER}"

    def test_doctor_sentence_from_db_fields(self):
        reply = fr.build_fallback_reply("symptom", [], _doctor())
        assert "A Dermatologist can look into this." in reply
        assert "Dr. Sana Malik (Dermatologist, 9 years of experience)" in reply
        assert reply.endswith(SAFETY_DISCLAIMER)

    def test_placeholder_name_is_not_repeated(self):
        doctor = _doctor()
        doctor.doctor_name = "Dr. Dermatologist"
        reply = fr.build_fallback_reply("symptom", None, doctor)
        assert "Dr. Dermatologist" not in reply
        assert "The suggested Dermatologist has 9 years of experience." in reply

    def test_requested_specialist_is_explained(self):
        doctor = _doctor(requested="Orthopedic Surgeon")
        doctor.specialist = "General Physician"
        reply = fr.build_fallback_reply("symptom", None, doctor)
        assert "No Orthopedic Surgeon is available right now" in reply
        assert "General Physician" in reply

    def test_faq_and_doctor_both_used(self):
        reply = fr.build_fallback_reply("general", [_doc()], _doctor())
        assert reply.index(FAQ_ANSWER) < reply.index("Dermatologist")

    def test_doctor_dict_dump_works(self):
        reply = fr.build_fallback_reply("symptom", None, _doctor().model_dump())
        assert "Dr. Sana Malik" in reply

    @pytest.mark.parametrize(
        "intent,expected",
        [
            ("general", fr.FALLBACK_GENERAL),
            ("booking", fr.FALLBACK_BOOKING),
            ("symptom", fr.FALLBACK_SYMPTOM),
            ("medication", fr.FALLBACK_MEDICATION),
            ("unknown", fr.FALLBACK_GENERAL),
        ],
    )
    def test_nothing_retrieved_uses_static_text(self, intent, expected):
        assert fr.build_fallback_reply(intent, [], None) == expected

    def test_disclaimer_appears_once(self):
        reply = fr.build_fallback_reply("general", [_doc()], _doctor())
        assert reply.count(SAFETY_DISCLAIMER) == 1

    @pytest.mark.parametrize("faq", FAQS, ids=lambda f: f["question"][:40])
    def test_every_seeded_faq_fallback_passes_guardrails(self, faq):
        reply = fr.build_fallback_reply("general", [_doc(faq["answer"])], None)
        is_safe, violations = SafetyGuardrails.validate_response(reply)
        assert is_safe, violations

    @pytest.mark.parametrize(
        "text",
        [
            fr.FALLBACK_GENERAL,
            fr.FALLBACK_SYMPTOM,
            fr.FALLBACK_MEDICATION,
            fr.FALLBACK_BOOKING,
            fr.FALLBACK_EMERGENCY,
            fr.build_fallback_reply("symptom", None, _doctor()),
            fr.build_fallback_reply("symptom", None, _doctor("Cardiologist")),
        ],
    )
    def test_static_and_doctor_texts_pass_guardrails(self, text):
        assert SafetyGuardrails.validate_response(text)[0]


# ---------------------------------------------------------------------------
# Chatbot reply paths
# ---------------------------------------------------------------------------
class _FakeRag:
    def __init__(self, docs: List[Any]):
        self.docs = docs

    def retrieve(self, query, threshold=None):
        return list(self.docs)


class _FakeIntent:
    def __init__(self, intent: IntentEnum):
        self.intent = intent

    async def classify_intent(self, message):
        return (self.intent, 0.95)


def _bot(intent=IntentEnum.GENERAL, docs=None, doctor=None) -> HokuChatbot:
    bot = HokuChatbot()
    bot._rag = _FakeRag(docs or [])
    bot._intent_classifier = _FakeIntent(intent)
    bot._lookup_doctor_suggestion = AsyncMock(return_value=doctor)  # type: ignore[method-assign]
    return bot


@pytest.fixture
def no_memory():
    with patch("app.ai.chatbot.HokuConversationMemory") as mem:
        mem.return_value.load_memory.return_value = []
        yield


def _no_llm(bot):
    return patch.object(
        type(bot), "main_llm", new_callable=lambda: property(lambda self: None)
    )


class TestChatbotFallbackPaths:
    async def test_no_llm_uses_faq_answer(self, no_memory):
        bot = _bot(docs=[_doc()])
        with _no_llm(bot):
            result = await bot.get_response("home visit wait?", 1, MagicMock())
        assert result["fallback_used"] is True
        assert result["reply"] == f"{FAQ_ANSWER} {SAFETY_DISCLAIMER}"
        assert result["intent"] == "general"
        assert result["confidence"] == 0.95

    async def test_no_llm_keeps_doctor_suggestion(self, no_memory):
        bot = _bot(intent=IntentEnum.SYMPTOM, doctor=_doctor())
        with _no_llm(bot):
            result = await bot.get_response("itchy rash for days", 1, MagicMock())
        assert result["fallback_used"] is True
        assert result["doctor_suggestion"]["doctor_name"] == "Dr. Sana Malik"
        assert result["suggestedSpecialist"] == "Dermatologist"
        assert "Dr. Sana Malik" in result["reply"]

    async def test_no_llm_nothing_retrieved_static(self, no_memory):
        bot = _bot(intent=IntentEnum.MEDICATION)
        with _no_llm(bot):
            result = await bot.get_response("can I double my pills", 1, MagicMock())
        assert result["reply"] == fr.FALLBACK_MEDICATION
        assert result["fallback_used"] is True

    async def test_booking_intent_is_grounded_on_faq(self, no_memory):
        bot = _bot(intent=IntentEnum.BOOKING, docs=[_doc()])
        with _no_llm(bot):
            result = await bot.get_response("how long to get a visit", 1, MagicMock())
        assert FAQ_ANSWER in result["reply"]

    async def test_medication_intent_discards_faq(self, no_memory):
        bot = _bot(intent=IntentEnum.MEDICATION, docs=[_doc()])
        with _no_llm(bot):
            result = await bot.get_response("medicine question", 1, MagicMock())
        assert FAQ_ANSWER not in result["reply"]

    async def test_llm_timeout_uses_retrieved_content(self, no_memory):
        bot = _bot(docs=[_doc()])
        bot._main_llm = MagicMock()
        with patch("app.ai.chatbot.HokuChatbot._generation_chain"), patch(
            "app.ai.chatbot.generate_with_timeout", AsyncMock(return_value=None)
        ):
            result = await bot.get_response("wait time?", 1, MagicMock())
        assert result["fallback_used"] is True
        assert FAQ_ANSWER in result["reply"]

    async def test_llm_error_uses_retrieved_content(self, no_memory):
        bot = _bot(intent=IntentEnum.SYMPTOM, doctor=_doctor())
        bot._main_llm = MagicMock()
        with patch("app.ai.chatbot.HokuChatbot._generation_chain"), patch(
            "app.ai.chatbot.generate_with_timeout",
            AsyncMock(side_effect=RuntimeError("provider down")),
        ):
            result = await bot.get_response("rash", 1, MagicMock())
        assert result["fallback_used"] is True
        assert result["doctor_suggestion"]["specialist"] == "Dermatologist"

    async def test_rag_exception_means_no_faq(self, no_memory):
        bot = _bot()

        class _Broken:
            def retrieve(self, query, threshold=None):
                raise RuntimeError("db down")

        bot._rag = _Broken()
        with _no_llm(bot):
            result = await bot.get_response("services?", 1, MagicMock())
        assert result["reply"] == fr.FALLBACK_GENERAL

    async def test_llm_success_is_not_fallback(self, no_memory):
        bot = _bot(docs=[_doc()])
        bot._main_llm = MagicMock()
        payload = json.dumps(
            {
                "reply": f"Visits are usually confirmed within hours. {SAFETY_DISCLAIMER}",
                "suggestedSpecialist": None,
                "severity": "mild",
                "shouldSeeDoctor": False,
            }
        )
        with patch("app.ai.chatbot.HokuChatbot._generation_chain"), patch(
            "app.ai.chatbot.generate_with_timeout", AsyncMock(return_value=payload)
        ):
            result = await bot.get_response("wait time?", 1, MagicMock())
        assert result["fallback_used"] is False
        assert result["reply"].startswith("Visits are usually confirmed")


# ---------------------------------------------------------------------------
# API field
# ---------------------------------------------------------------------------
def test_chat_api_returns_fallback_used(client, auth_headers):
    resp = client.post(
        "/api/ai/chat", json={"message": "hello", "userId": 1}, headers=auth_headers
    )
    assert resp.status_code == 200
    assert isinstance(resp.json()["fallback_used"], bool)


def test_emergency_reply_is_not_marked_fallback(client, auth_headers):
    resp = client.post(
        "/api/ai/chat",
        json={"message": "I have severe chest pain and can't breathe", "userId": 1},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Hoku-Emergency") == "true"
    assert resp.json()["fallback_used"] is False


# ---------------------------------------------------------------------------
# Prompt rules
# ---------------------------------------------------------------------------
class TestReplyRules:
    @pytest.mark.parametrize("prompt", [SYSTEM_PROMPT, RAG_SYSTEM_PROMPT])
    def test_rules_in_both_system_prompts(self, prompt):
        assert REPLY_RULES in prompt
        lowered = prompt.lower()
        assert "plain text" in lowered
        assert "do not repeat information as a list" in lowered
        assert "do not promise anything you cannot do" in lowered
        assert "answer only health questions" in lowered

    def test_rules_reach_the_rendered_prompt_on_both_paths(self):
        plain = chat_prompt_template.format_messages(
            message="m", context="c", history=[]
        )
        rag = rag_chat_prompt_template.format_messages(
            message="m", context="c", history=[], faq_context="- Q: q\n  A: a"
        )
        for messages in (plain, rag):
            assert REPLY_RULES.strip() in messages[0].content
