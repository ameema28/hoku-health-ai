"""
Hoku Health Care - Application Constants.

Centralized constants to avoid magic strings and ensure consistency
across the AI chatbot and related services.
"""

from typing import Any, Optional

# Input Constraints
MAX_MESSAGE_LENGTH: int = 1000

# Clinical Safety
# Closing line of health, symptom, medication and emergency replies.
SAFETY_DISCLAIMER: str = "Please consult a doctor for proper diagnosis."
# Closing line of service, booking and general FAQ replies.
GENERAL_DISCLAIMER: str = "General information only, not medical advice."
# Closing line of the fixed self-harm reply (replaces SAFETY_DISCLAIMER there).
SELF_HARM_CLOSING: str = (
    "You can also talk to a doctor or a mental-health professional."
)

_ALL_CLOSINGS = (SAFETY_DISCLAIMER, GENERAL_DISCLAIMER, SELF_HARM_CLOSING)

# Intents whose replies are health advice (health disclaimer, "See a doctor").
HEALTH_INTENTS = frozenset({"symptom", "medication", "emergency"})


def is_health_reply(
    intent: Optional[str],
    doctor_suggestion: Any = None,
    symptom_wording: bool = False,
) -> bool:
    """
    True if a reply is about health: a health intent, a suggested doctor, or
    symptom wording in the message (a general-intent question such as
    "I have a headache" when the classifier was unsure or unavailable).
    """
    return (
        (intent or "").lower() in HEALTH_INTENTS
        or doctor_suggestion is not None
        or symptom_wording
    )


def has_disclaimer(text: str) -> bool:
    """True if ``text`` contains any of the accepted closing disclaimers."""
    lowered = (text or "").lower()
    return any(closing.lower() in lowered for closing in _ALL_CLOSINGS)


def apply_disclaimer(text: str, health: bool = True) -> str:
    """
    End ``text`` with the disclaimer for its context.

    Health replies end with ``SAFETY_DISCLAIMER``; other replies end with
    ``GENERAL_DISCLAIMER``. The other context's disclaimer is removed, and a
    reply that already ends with the right one is returned unchanged. A
    reply that ends with the self-harm closing is left as it is.
    """
    text = (text or "").strip()
    if text.endswith(SELF_HARM_CLOSING):
        return text
    wanted, other = (
        (SAFETY_DISCLAIMER, GENERAL_DISCLAIMER)
        if health
        else (GENERAL_DISCLAIMER, SAFETY_DISCLAIMER)
    )
    text = text.replace(other, "").strip()
    if wanted in text:
        return text
    return f"{text} {wanted}".strip()
