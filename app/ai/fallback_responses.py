"""
Hoku Health Care - Deterministic fallback replies.

Used when the reply LLM is unavailable, times out, errors or returns an
empty reply. The text is built only from content the request already
retrieved: the top matching FAQ answer and the doctor suggestion from the
database. Nothing is generated or invented here. When nothing was retrieved,
a static intent-specific message is used. Every reply ends with the
mandatory disclaimer, and the chatbot still runs the safety guardrails on
the result.

(Emergency messages never reach this module: they are answered by the
emergency detector before any LLM call.)
"""

from typing import Any, Optional, Sequence

from app.utils.constants import SAFETY_DISCLAIMER

# ------------------------------------------------------------------
# Static messages (nothing retrieved for the request)
# ------------------------------------------------------------------
FALLBACK_GENERAL: str = (
    "I can't give a full answer right now. I can help with questions about "
    "Hoku Health Care's home-care services, booking a visit, and general "
    f"health information. {SAFETY_DISCLAIMER}"
)

FALLBACK_SYMPTOM: str = (
    "I can't give a full answer right now. If your symptoms are severe or "
    "getting worse, contact a doctor or your local emergency number. "
    f"{SAFETY_DISCLAIMER}"
)

FALLBACK_MEDICATION: str = (
    "I can't advise on medication. Please ask your doctor or pharmacist. "
    f"{SAFETY_DISCLAIMER}"
)

FALLBACK_BOOKING: str = (
    "You can book an appointment directly through the patient dashboard. "
    f"{SAFETY_DISCLAIMER}"
)

FALLBACK_EMERGENCY: str = (
    "This may be serious. Please contact emergency services immediately "
    f"or visit the nearest ER. {SAFETY_DISCLAIMER}"
)


def get_fallback_for_intent(intent: str) -> str:
    """
    Static fallback for an intent when nothing was retrieved.

    Args:
        intent: Classified intent string (e.g., "booking", "symptom").

    Returns:
        str: The matching message, defaulting to FALLBACK_GENERAL.
    """
    intent_lower = (intent or "").lower()
    if intent_lower == "emergency":
        return FALLBACK_EMERGENCY
    if intent_lower == "booking":
        return FALLBACK_BOOKING
    if intent_lower == "symptom":
        return FALLBACK_SYMPTOM
    if intent_lower == "medication":
        return FALLBACK_MEDICATION
    return FALLBACK_GENERAL


def _field(obj: Any, name: str) -> Any:
    """Read a field from a DoctorSuggestion model or its dict dump."""
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def doctor_sentence(doctor_suggestion: Any) -> str:
    """
    One sentence about the suggested doctor, from database fields only.

    The demo page and API clients show the full suggestion (including the
    available slots) from ``doctor_suggestion``, so the text does not
    repeat the slot list.
    """
    specialist = _field(doctor_suggestion, "specialist")
    if not specialist:
        return ""
    name = _field(doctor_suggestion, "doctor_name")
    experience = _field(doctor_suggestion, "experience")
    requested = _field(doctor_suggestion, "requested_specialist")

    parts = []
    if requested:
        parts.append(
            f"No {requested} is available right now, so a {specialist} is "
            "suggested as a first contact."
        )
    else:
        parts.append(f"A {specialist} can look into this.")
    if name:
        detail = f"{name} ({specialist}"
        if isinstance(experience, int) and experience > 0:
            detail += f", {experience} years of experience"
        parts.append(f"Suggested doctor: {detail}).")
    return " ".join(parts)


def build_fallback_reply(
    intent: str,
    faq_docs: Optional[Sequence[Any]] = None,
    doctor_suggestion: Any = None,
) -> str:
    """
    Return a deterministic reply from content retrieved for this request.

    Order: the top FAQ answer, then the doctor sentence. If neither exists,
    the static message for the intent. Always ends with the disclaimer.

    Args:
        intent: Classified intent value.
        faq_docs: FAQ documents that passed the retrieval threshold
            (``metadata["answer"]`` is used from the first one).
        doctor_suggestion: DoctorSuggestion (or its dict) from the DB lookup.
    """
    parts = []
    if faq_docs:
        answer = str(getattr(faq_docs[0], "metadata", {}).get("answer", "")).strip()
        if answer:
            parts.append(answer)
    if doctor_suggestion is not None:
        sentence = doctor_sentence(doctor_suggestion)
        if sentence:
            parts.append(sentence)

    if not parts:
        return get_fallback_for_intent(intent)

    parts.append(SAFETY_DISCLAIMER)
    return " ".join(parts)
