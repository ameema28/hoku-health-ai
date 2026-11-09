"""
A reply that states a medicine dose, amount or dosing interval is replaced
with the fixed medication message (no rewriting). No LLM, no network.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.ai.fallback_responses import FALLBACK_MEDICATION
from app.ai.prompts import REPLY_RULES
from app.ai.safety_guardrails import SafetyGuardrails, contains_dose
from app.utils.constants import SAFETY_DISCLAIMER


@pytest.mark.parametrize(
    "text",
    [
        "The usual adult dose of paracetamol is 500 mg to 1 g every 4 to 6 hours.",
        "Adults can take 2 tablets of ibuprofen three times a day with food.",
        "You can give 5ml of the syrup.",
        "Take one capsule every 8 hours.",
        "Use the inhaler twice a day.",
        "Vitamin D 1000 IU is common.",
    ],
)
def test_dose_replaced_with_medication_fallback(text):
    result, violations, severity = SafetyGuardrails.apply_3_strike_safety(
        text=f"{text} {SAFETY_DISCLAIMER}", user_id=1, db=None
    )
    assert result == FALLBACK_MEDICATION
    assert violations == [SafetyGuardrails.VIOLATION_PRESCRIPTION]
    assert severity == "high"


@pytest.mark.parametrize(
    "text",
    [
        "Wash the area gently twice a day and keep it dry.",
        "Most home visits are confirmed within 24 hours.",
        "In Pakistan, call 1122 for an ambulance.",
        "Drink plenty of water and rest for a few days.",
        "A doctor or pharmacist can advise on the right dose for you.",
    ],
)
def test_benign_text_is_not_a_dose(text):
    assert contains_dose(text) is False


def test_dose_event_is_logged_without_reply_text():
    db = MagicMock()
    with patch("app.ai.safety_guardrails.log_safety_violation") as log_event:
        SafetyGuardrails.apply_3_strike_safety(
            text=f"Take 400 mg now. {SAFETY_DISCLAIMER}", user_id=7, db=db
        )
    kwargs = log_event.call_args.kwargs
    assert kwargs["violation_type"] == "prescription_advice"
    assert "400" not in kwargs["ai_response"]


def test_no_seeded_faq_answer_is_withheld_as_a_dose():
    """FAQ fallback replies pass through the same guard."""
    from app.scripts.seed_faqs import FAQS

    flagged = [f["question"] for f in FAQS if contains_dose(f["answer"])]
    assert flagged == []


def test_reply_rules_forbid_doses():
    assert "never give a dose" in REPLY_RULES.lower()
