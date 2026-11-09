"""
Hoku Health Care - Emergency Detection Module.

Runs BEFORE any LLM call: a deterministic, auditable keyword check (regex,
well under 50 ms). If it fires, the chatbot returns a fixed urgent reply and
skips the LLM, RAG, cache and doctor lookup.

A second signal comes from the intent classifier: when it labels a message
``emergency``, the chatbot answers with the same fixed reply (see
``HokuChatbot.get_response``). That covers phrasings the keyword list misses,
on a best-effort basis.

Coverage limits:
- The keyword list is English only. Messages in other languages (for
  example Urdu or Arabic) rely on the intent classifier.
- Matching is on whole words and phrases. Curly apostrophes (as typed by
  phone keyboards) are normalised to ``'`` first, and "cant" is accepted.

Urgency levels: ``self_harm``, ``high``, ``moderate`` (and ``none``).
"""

import logging
import re
from typing import Any, Dict, List, Tuple

from app.utils.constants import SAFETY_DISCLAIMER, SELF_HARM_CLOSING

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# RED FLAG PHRASES (regex, whole words, case-insensitive)
# ------------------------------------------------------------------
# The lists are conservative: they bias toward flagging rather than
# missing. Apostrophes are normalised before matching (see _normalise).

# Self-harm or suicide: a separate reply with crisis wording.
_RED_FLAG_SELF_HARM: List[str] = [
    "suicide",
    "suicidal",
    "kill myself",
    "killing myself",
    "end my life",
    "ending my life",
    "take my own life",
    "want to die",
    "don't want to live",
    "dont want to live",
    "hurt myself",
    "harm myself",
    "self-harm",
    "self harm",
]

# High urgency: immediate emergency services required.
_RED_FLAG_HIGH: List[str] = [
    "chest pain",
    "can't breathe",
    "cant breathe",
    "cannot breathe",
    "can not breathe",
    "not breathing",
    "difficulty breathing",
    "trouble breathing",
    "struggling to breathe",
    "short of breath",
    "shortness of breath",
    "severe bleeding",
    "bleeding heavily",
    "unconscious",
    "unresponsive",
    "not responding",
    "isn't responding",
    "collapsed",
    "heart attack",
    "cardiac arrest",
    "stroke",
    "seizure",
    "anaphylaxis",
    "overdose",
    "overdosed",
    "poisoning",
    "severe allergic reaction",
    "choking",
    "drowning",
    "major trauma",
    "head injury",
    "passed out",
    "fainting",
    "blue lips",
    "lips are blue",
    "lips turning blue",
    "blue face",
    "cold sweat",
    "crushing chest",
    "chest is crushing",
    "can't move",
    "cant move",
    "cannot move",
    "paralyzed",
    "slurred speech",
    "one side weak",
    "facial drooping",
    "worst headache",
    "thunderclap headache",
]

# Moderate urgency: urgent but not immediately life-threatening. Generic
# words that are common in ordinary questions ("dehydration", "confusion")
# are only matched in their urgent forms.
_RED_FLAG_MODERATE: List[str] = [
    "high fever",
    "severe dehydration",
    "persistent vomiting",
    "unable to urinate",
    "sudden confusion",
    "altered mental state",
]


def _compile(phrases: List[str]) -> "re.Pattern[str]":
    return re.compile(
        r"\b(" + "|".join(re.escape(p) for p in phrases) + r")\b",
        re.IGNORECASE,
    )


_SELF_HARM_PATTERN = _compile(_RED_FLAG_SELF_HARM)
_HIGH_URGENCY_PATTERN = _compile(_RED_FLAG_HIGH)
_MODERATE_URGENCY_PATTERN = _compile(_RED_FLAG_MODERATE)

# Curly and modifier apostrophes typed by phone keyboards.
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def _normalise(message: str) -> str:
    """Map typographic apostrophes to ``'`` so "can’t" matches "can't"."""
    return message.translate(_APOSTROPHES)


class EmergencyDetector:
    """
    Hoku Health Care emergency detection (keyword check).

    All methods are classmethods for stateless operation.
    """

    URGENCY_SELF_HARM = "self_harm"
    URGENCY_HIGH = "high"
    URGENCY_MODERATE = "moderate"
    URGENCY_NONE = "none"

    # Fixed urgent replies. No phone number is given for crisis lines:
    # only numbers we can stand behind are listed.
    URGENCY_RESPONSES: Dict[str, Dict[str, Any]] = {
        URGENCY_SELF_HARM: {
            "reply": (
                "I'm really sorry you're going through this. You don't have to "
                "face it alone.\n\n"
                "If you might act on these thoughts, or you are in danger, please "
                "contact your local emergency number or a crisis line in your "
                "country now. If you can, tell someone near you how you are "
                "feeling.\n\n"
                f"{SELF_HARM_CLOSING}"
            ),
            "suggestedSpecialist": None,
            "severity": "severe",
            "shouldSeeDoctor": True,
            "intent": "emergency",
            "confidence": 1.0,
        },
        URGENCY_HIGH: {
            "reply": (
                "🚨 This could be an emergency 🚨\n\n"
                "This may be a life-threatening situation. "
                "Please call your local emergency number now:\n\n"
                "• Pakistan: 1122 (Rescue 1122) or 15 (Police Emergency)\n"
                "• UAE: 998 (Ambulance) or 999 (Police)\n"
                "• UK: 999 (Emergency) or 111 (NHS Non-emergency)\n\n"
                "Do not wait for a chatbot response. Seek immediate medical help. "
                f"{SAFETY_DISCLAIMER}"
            ),
            "suggestedSpecialist": "Emergency Medicine",
            "severity": "severe",
            "shouldSeeDoctor": True,
            "intent": "emergency",
            "confidence": 1.0,
        },
        URGENCY_MODERATE: {
            "reply": (
                "⚠️ These symptoms may need urgent attention ⚠️\n\n"
                "Please contact a healthcare provider or visit an urgent care "
                "facility as soon as possible. If things get worse, call your "
                "local emergency number:\n\n"
                "• Pakistan: 1122 (Rescue 1122)\n"
                "• UAE: 998 (Ambulance) or 999 (Police)\n"
                "• UK: 111 (NHS Non-emergency) or 999 (Emergency)\n\n"
                f"{SAFETY_DISCLAIMER}"
            ),
            "suggestedSpecialist": "General Physician",
            "severity": "moderate",
            "shouldSeeDoctor": True,
            "intent": "emergency",
            "confidence": 1.0,
        },
    }

    @classmethod
    def detect_emergency(cls, message: str) -> Tuple[bool, str, str]:
        """
        Detect red-flag phrases with pre-compiled regexes (no LLM).

        Checked in order: self-harm, high, moderate.

        Args:
            message: The raw user message.

        Returns:
            Tuple[bool, str, str]: (is_emergency, urgency_level, matched
            phrase). The matched phrase is patient text: never log it.
        """
        if not message or not isinstance(message, str):
            return (False, cls.URGENCY_NONE, "")

        text = _normalise(message)
        for pattern, urgency in (
            (_SELF_HARM_PATTERN, cls.URGENCY_SELF_HARM),
            (_HIGH_URGENCY_PATTERN, cls.URGENCY_HIGH),
            (_MODERATE_URGENCY_PATTERN, cls.URGENCY_MODERATE),
        ):
            match = pattern.search(text)
            if match:
                # Never log the message or the matched words (patient text).
                logger.warning(
                    "Emergency phrase detected (urgency=%s, message_len=%d)",
                    urgency,
                    len(message),
                )
                return (True, urgency, match.group(0))

        return (False, cls.URGENCY_NONE, "")

    @classmethod
    def get_urgency_response(cls, urgency_level: str) -> Dict[str, Any]:
        """
        Return a copy of the fixed reply for ``urgency_level``.

        Unknown levels get the high-urgency reply (safety bias).
        """
        response = cls.URGENCY_RESPONSES.get(urgency_level)
        if response is None:
            logger.warning(
                "Unknown urgency_level='%s', defaulting to high", urgency_level
            )
            response = cls.URGENCY_RESPONSES[cls.URGENCY_HIGH]
        return dict(response)


# ------------------------------------------------------------------
# Module-level helpers (used by tests and app.ai.__init__)
# ------------------------------------------------------------------


def detect_emergency(message: str) -> bool:
    """True if any red-flag phrase (any urgency) is in ``message``."""
    is_emergency, _, _ = EmergencyDetector.detect_emergency(message)
    return is_emergency


def get_emergency_response() -> Dict[str, Any]:
    """The high-urgency fixed reply."""
    return EmergencyDetector.get_urgency_response(EmergencyDetector.URGENCY_HIGH)
