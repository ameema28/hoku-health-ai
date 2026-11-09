"""
Demo polish: context disclaimer, shouldSeeDoctor only for health replies,
gentler self-harm reply, and the demo page wording. All LLM calls are mocked.
"""

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai import fallback_responses as fr
from app.ai.emergency_detector import EmergencyDetector
from app.ai.intent_classifier import IntentEnum
from app.ai.safety_guardrails import SafetyGuardrails
from app.demo_page import DEMO_PAGE_HTML
from app.utils.constants import (
    GENERAL_DISCLAIMER,
    SAFETY_DISCLAIMER,
    SELF_HARM_CLOSING,
    apply_disclaimer,
    has_disclaimer,
    is_health_reply,
)
from tests.test_fallback_replies import _bot, _doc, _doctor, no_memory  # noqa: F401


def _script() -> str:
    return re.search(r"<script>(.*?)</script>", DEMO_PAGE_HTML, re.DOTALL).group(1)


# ---------------------------------------------------------------------------
# Item 2: disclaimer by context
# ---------------------------------------------------------------------------
class TestDisclaimerHelpers:
    def test_constants(self):
        assert SAFETY_DISCLAIMER == "Please consult a doctor for proper diagnosis."
        assert GENERAL_DISCLAIMER == "General information only, not medical advice."

    def test_health_reply_gets_medical_disclaimer(self):
        assert apply_disclaimer("Rest.", health=True) == f"Rest. {SAFETY_DISCLAIMER}"

    def test_service_reply_gets_general_disclaimer(self):
        assert apply_disclaimer("We help.", health=False) == (
            f"We help. {GENERAL_DISCLAIMER}"
        )

    def test_wrong_disclaimer_is_swapped_not_doubled(self):
        out = apply_disclaimer(f"We help. {SAFETY_DISCLAIMER}", health=False)
        assert out == f"We help. {GENERAL_DISCLAIMER}"
        out = apply_disclaimer(f"Rest. {GENERAL_DISCLAIMER}", health=True)
        assert out == f"Rest. {SAFETY_DISCLAIMER}"

    def test_right_disclaimer_is_kept_once(self):
        text = f"Rest. {SAFETY_DISCLAIMER}"
        assert apply_disclaimer(text, health=True) == text

    def test_self_harm_closing_is_left_alone(self):
        text = f"I'm here. {SELF_HARM_CLOSING}"
        assert apply_disclaimer(text, health=True) == text

    def test_has_disclaimer_accepts_all_three(self):
        for closing in (SAFETY_DISCLAIMER, GENERAL_DISCLAIMER, SELF_HARM_CLOSING):
            assert has_disclaimer(f"x {closing}")
        assert not has_disclaimer("no closing line")

    def test_is_health_reply(self):
        for intent in ("symptom", "medication", "emergency"):
            assert is_health_reply(intent)
        assert not is_health_reply("booking")
        assert not is_health_reply("general")
        assert is_health_reply("general", doctor_suggestion=object())
        assert is_health_reply("general", symptom_wording=True)

    def test_guardrails_accept_the_general_disclaimer(self):
        text = f"We offer home nursing in Lahore. {GENERAL_DISCLAIMER}"
        assert SafetyGuardrails.validate_response(text)[0]
        assert SafetyGuardrails.add_disclaimer(text) == text

    def test_guardrails_still_require_some_disclaimer(self):
        ok, violations = SafetyGuardrails.validate_response("We offer nursing.")
        assert not ok
        assert SafetyGuardrails.VIOLATION_MISSING_DISCLAIMER in violations


class TestFallbackDisclaimers:
    def test_faq_fallback_ends_with_general_disclaimer(self):
        assert fr.build_fallback_reply("general", [_doc()], None).endswith(
            GENERAL_DISCLAIMER
        )

    def test_doctor_fallback_ends_with_medical_disclaimer(self):
        assert fr.build_fallback_reply("symptom", None, _doctor()).endswith(
            SAFETY_DISCLAIMER
        )

    def test_static_texts(self):
        assert fr.FALLBACK_GENERAL.endswith(GENERAL_DISCLAIMER)
        assert fr.FALLBACK_BOOKING.endswith(GENERAL_DISCLAIMER)
        for text in (
            fr.FALLBACK_SYMPTOM,
            fr.FALLBACK_MEDICATION,
            fr.FALLBACK_EMERGENCY,
        ):
            assert text.endswith(SAFETY_DISCLAIMER)


def _llm_payload(reply: str, should_see: bool = True) -> str:
    return json.dumps(
        {
            "reply": reply,
            "suggestedSpecialist": None,
            "severity": "mild",
            "shouldSeeDoctor": should_see,
        }
    )


async def _llm_reply(intent, message, doctor=None, reply="Here is the answer."):
    bot = _bot(intent=intent, doctor=doctor)
    bot._main_llm = MagicMock()
    generate = AsyncMock(return_value=_llm_payload(reply))
    with patch("app.ai.chatbot.HokuChatbot._generation_chain"), patch(
        "app.ai.chatbot.generate_with_timeout", generate
    ):
        return await bot.get_response(message, 1, MagicMock())


class TestLlmReplyContext:
    async def test_booking_reply_gets_general_disclaimer(self, no_memory):  # noqa: F811
        # The model followed the old prompt rule and added the medical line.
        result = await _llm_reply(
            IntentEnum.BOOKING,
            "How do I book a visit?",
            reply=f"Booking is not part of this demo. {SAFETY_DISCLAIMER}",
        )
        assert result["reply"].endswith(GENERAL_DISCLAIMER)
        assert SAFETY_DISCLAIMER not in result["reply"]
        assert result["shouldSeeDoctor"] is False

    async def test_service_faq_reply_gets_general_disclaimer(self, no_memory):  # noqa: F811
        result = await _llm_reply(IntentEnum.GENERAL, "Do you offer home nursing?")
        assert result["reply"].endswith(GENERAL_DISCLAIMER)
        # The model said true; a non-health reply is forced to false.
        assert result["shouldSeeDoctor"] is False

    async def test_symptom_reply_keeps_medical_disclaimer(self, no_memory):  # noqa: F811
        result = await _llm_reply(
            IntentEnum.SYMPTOM, "I have had a rash for a week", doctor=_doctor()
        )
        assert result["reply"].endswith(SAFETY_DISCLAIMER)
        assert result["shouldSeeDoctor"] is True

    async def test_medication_reply_keeps_medical_disclaimer(self, no_memory):  # noqa: F811
        result = await _llm_reply(IntentEnum.MEDICATION, "Can I take aspirin?")
        assert result["reply"].endswith(SAFETY_DISCLAIMER)
        assert result["shouldSeeDoctor"] is True

    async def test_general_message_with_symptom_words_is_health(self, no_memory):  # noqa: F811
        result = await _llm_reply(IntentEnum.GENERAL, "I have a headache")
        assert result["reply"].endswith(SAFETY_DISCLAIMER)
        assert result["shouldSeeDoctor"] is True

    async def test_fallback_for_booking_is_not_see_doctor(self, no_memory):  # noqa: F811
        bot = _bot(intent=IntentEnum.BOOKING)
        with patch.object(
            type(bot), "main_llm", new_callable=lambda: property(lambda self: None)
        ):
            result = await bot.get_response("Can I book a visit?", 1, MagicMock())
        assert result["fallback_used"] is True
        assert result["reply"].endswith(GENERAL_DISCLAIMER)
        assert result["shouldSeeDoctor"] is False


# ---------------------------------------------------------------------------
# Item 5: self-harm reply
# ---------------------------------------------------------------------------
class TestSelfHarmReply:
    def test_closing_line_replaces_the_diagnosis_line(self):
        reply = EmergencyDetector.get_urgency_response("self_harm")["reply"]
        assert reply.endswith(
            "You can also talk to a doctor or a mental-health professional."
        )
        assert SAFETY_DISCLAIMER not in reply
        assert "crisis line in your country" in reply
        assert not any(ch.isdigit() for ch in reply)

    def test_other_emergencies_keep_the_medical_disclaimer(self):
        for level in ("high", "moderate"):
            reply = EmergencyDetector.get_urgency_response(level)["reply"]
            assert reply.endswith(SAFETY_DISCLAIMER)

    def test_endpoint_marks_self_harm_and_keeps_the_closing(self, client, auth_headers):
        resp = client.post(
            "/api/ai/chat",
            json={"message": "I want to kill myself", "userId": 1},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.headers["X-Hoku-Emergency"] == "true"
        assert resp.headers["X-Hoku-Emergency-Kind"] == "self_harm"
        body = resp.json()
        assert body["reply"].endswith(SELF_HARM_CLOSING)
        assert SAFETY_DISCLAIMER not in body["reply"]
        assert "emergency_kind" not in body

    def test_endpoint_marks_other_emergencies(self, client, auth_headers):
        resp = client.post(
            "/api/ai/chat",
            json={"message": "I have severe chest pain and can't breathe", "userId": 1},
            headers=auth_headers,
        )
        assert resp.headers["X-Hoku-Emergency-Kind"] == "high"
        assert resp.json()["reply"].endswith(SAFETY_DISCLAIMER)

    def test_non_emergency_has_no_kind_header(self, client, auth_headers):
        resp = client.post(
            "/api/ai/chat",
            json={"message": "hello", "userId": 1},
            headers=auth_headers,
        )
        assert "X-Hoku-Emergency-Kind" not in resp.headers


# ---------------------------------------------------------------------------
# Items 1, 3, 4, 5: demo page
# ---------------------------------------------------------------------------
class TestDemoPageWording:
    def test_fixed_emergency_fallback_label(self):
        script = _script()
        assert "no (fixed emergency reply, no AI model used)" in script
        # The label is chosen from the emergency header, not the API flag.
        assert "fallbackText(data.fallback_used, emergency)" in script
        assert "no (live model)" in script  # still used for normal replies

    def test_see_doctor_is_na_outside_health(self):
        script = _script()
        assert '"n/a"' in script
        assert '["See a doctor", seeDoctorText(data)]' in script
        assert '"symptom", "medication", "emergency"' in script

    def test_self_harm_header_is_not_the_red_emergency_label(self):
        script = _script()
        assert "You're not alone" in script
        assert 'headers.get("X-Hoku-Emergency-Kind") === "self_harm"' in script
        assert ".msg.support" in DEMO_PAGE_HTML
        assert "Possible emergency" in script  # other emergencies unchanged

    def test_compact_header_on_small_screens(self):
        html = DEMO_PAGE_HTML
        assert "@media (max-width: 600px), (max-height: 640px)" in html
        assert "white-space: nowrap" in html
        assert 'class="title-short"' in html
        assert ".title-long { display: none; }" in html
        # The banner is shrunk, never hidden.
        banner_rules = re.findall(r"\.banner \{[^}]*\}", html)
        assert banner_rules
        assert not any("display: none" in rule for rule in banner_rules)
        assert "Not medical advice" in html


# ---------------------------------------------------------------------------
# OpenAPI wording: no product or safety claims the project cannot back up
# ---------------------------------------------------------------------------
class TestOpenApiWording:
    def test_description_is_a_demo_not_a_platform_claim(self, client):
        info = client.get("/openapi.json").json()["info"]
        description = info["description"]
        assert description.startswith("Demo API")
        assert "not medical advice" in description
        assert "platform" not in description.lower()
        assert "AI-powered" not in description
        assert "license" not in info  # no license is granted

    def test_chat_docs_make_no_safety_or_clinical_claim(self, client):
        op = client.get("/openapi.json").json()["paths"]["/api/ai/chat"]["post"]
        assert "non-diagnostic" in op["description"]
        assert "safe," not in op["description"]
        assert op["responses"]["200"]["description"] == "AI response with metadata."
