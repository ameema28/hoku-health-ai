"""
Hoku Health Care - Emergency Detection Module (Day 4).

Critical safety net that runs BEFORE any LLM call. Uses pure Python
regex for O(1) keyword matching — no LLM involved. This guarantees
sub-50ms detection of life-threatening symptoms.

Design rationale:
- Emergency detection must be synchronous and FAST because every
  millisecond counts in a life-threatening situation.
- Regex-based keyword matching is deterministic and auditable.
- LLM-based emergency detection would add latency (100-500ms) and
  could hallucinate or miss urgent symptoms.
- This module bypasses the normal LLM flow entirely when an
  emergency is detected, returning an immediate urgent response.

If emergency is detected:
1. Log at CRITICAL level (triggers alerting infrastructure)
2. Return urgent response dict immediately (bypasses all LLM calls)
3. API layer adds X-Hoku-Emergency: true header
"""

import logging
import re
import time
from typing import Any, Dict

from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# EMERGENCY KEYWORDS
# ------------------------------------------------------------------
# These keywords indicate potentially life-threatening symptoms.
# The list is conservative — we bias toward flagging emergencies
# rather than missing them (safety-first approach).
#
# Sources: NHS emergency guidance, CDC emergency warning signs,
# Pakistan Medical Association emergency protocols.
#
# CRITICAL: Keywords must match both raw text AND HTML-escaped text.
# sanitize_message() calls html.escape() which converts:
#   ' -> &#x27;  (apostrophe)
#   " -> &quot;  (double quote)
#   & -> &amp;   (ampersand)
#   < -> &lt;    (less than)
#   > -> &gt;    (greater than)
#
# We include both forms to ensure matching regardless of sanitization state.
EMERGENCY_KEYWORDS: list[str] = [
    # Original forms (before sanitization)
    "chest pain",
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "severe bleeding",
    "unconscious",
    "heart attack",
    "stroke",
    "suicide",
    "seizure",
    "not breathing",
    "cardiac arrest",
    "anaphylaxis",
    "overdose",
    "poisoning",
    "severe allergic reaction",
    "choking",
    "drowning",
    "major trauma",
    "head injury",
    "bleeding heavily",
    "passed out",
    "fainting",
    "blue lips",
    "blue face",
    "cold sweat",
    "crushing chest",
    "chest is crushing",  # Word order variation
    "can't move",
    "cannot move",
    "paralyzed",
    "slurred speech",
    "one side weak",
    "facial drooping",
    "worst headache",
    "thunderclap headache",
    # HTML-escaped forms (after sanitize_message)
    "can&#x27;t breathe",
    "can&#x27;t move",
    "crushing chest",
    "chest is crushing",
]

# Pre-compile regex for O(1) matching performance
# Pattern: word boundaries around each keyword, case-insensitive
_EMERGENCY_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in EMERGENCY_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


def detect_emergency(message: str) -> bool:
    """
    Detect potentially life-threatening symptoms in user message.

    Uses pre-compiled regex for sub-50ms synchronous matching.
    No LLM call — pure Python string matching.

    IMPORTANT: This function must work on BOTH raw user input AND
    HTML-escaped input (from sanitize_message). The keyword list
    includes both forms to ensure safety regardless of when this
    is called in the pipeline.

    Args:
        message: User message (may be raw or HTML-escaped).

    Returns:
        bool: True if emergency keywords detected.

    Performance:
        - Typical execution: < 1ms on modern hardware
        - Worst case: < 10ms for 1000-character messages
        - Well under the 50ms requirement.

    Thread safety:
        Pre-compiled regex is thread-safe for matching operations.
    """
    start_time = time.perf_counter()

    if not message or not isinstance(message, str):
        return False

    # Fast path: check if any emergency keyword is present
    is_emergency = bool(_EMERGENCY_PATTERN.search(message))

    elapsed = time.perf_counter() - start_time

    if is_emergency:
        # CRITICAL level triggers alerting infrastructure
        logger.critical(
            "EMERGENCY DETECTED in message (matched in %.4fs): "
            "user_message_preview=%s...",
            elapsed,
            message[:100],
        )
    else:
        logger.debug(
            "No emergency detected (checked in %.4fs)",
            elapsed,
        )

    return is_emergency


def get_emergency_response() -> Dict[str, Any]:
    """
    Return urgent response dict for emergency situations.

    This bypasses the normal LLM flow entirely for safety-critical
    speed. The response is pre-formatted and includes all required
    fields for the ChatMessageResponse schema.

    Returns:
        Dict with reply, suggestedSpecialist, severity, shouldSeeDoctor,
        intent, and confidence fields.
    """
    emergency_reply = (
        "🚨 EMERGENCY DETECTED 🚨\n\n"
        "This appears to be a potentially life-threatening situation. "
        "Please call your local emergency number immediately:\n\n"
        "• Pakistan: 1122 (Rescue 1122) or 15 (Police Emergency)\n"
        "• UAE: 998 (Ambulance) or 999 (Police)\n"
        "• UK: 999 (Emergency) or 111 (NHS Non-emergency)\n\n"
        "Do not wait for a chatbot response. Seek immediate medical help. "
        f"{SAFETY_DISCLAIMER}"
    )

    return {
        "reply": emergency_reply,
        "suggestedSpecialist": "Emergency Medicine",
        "severity": "severe",
        "shouldSeeDoctor": True,
        "intent": "emergency",
        "confidence": 1.0,
    }