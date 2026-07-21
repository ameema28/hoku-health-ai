"""
Hoku Health Care - Emergency Detection Module (Day 7).

Critical safety net that runs BEFORE any LLM call. Uses a two-tier
detection system:

Tier 1: Pure Python regex for O(1) keyword matching — no LLM involved.
    Guarantees sub-50ms detection of life-threatening symptoms.

Tier 2: Fast Groq LLM fallback (llama-3.1-8b-instant, 0.3s timeout)
    for ambiguous edge cases where Tier 1 is inconclusive.

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

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

from app.ai.config import ai_settings
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# TIER 1: RED FLAG SYMPTOMS (Regex-based, < 50ms)
# ------------------------------------------------------------------
# These keywords indicate potentially life-threatening symptoms.
# The list is conservative — we bias toward flagging emergencies
# rather than missing them (safety-first approach).
#
# Sources: NHS emergency guidance, CDC emergency warning signs,
# Pakistan Medical Association emergency protocols.

# High urgency: immediate emergency services required
_RED_FLAG_HIGH: list[str] = [
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
    "chest is crushing",
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
]

# Moderate urgency: urgent but not immediately life-threatening
_RED_FLAG_MODERATE: list[str] = [
    "high fever",
    "dehydration",
    "severe dehydration",
    "persistent vomiting",
    "severe abdominal pain",
    "unable to urinate",
    "confusion",
    "altered mental state",
]

# Pre-compile regex patterns for O(1) matching performance
_HIGH_URGENCY_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in _RED_FLAG_HIGH) + r')\b',
    re.IGNORECASE,
)

_MODERATE_URGENCY_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in _RED_FLAG_MODERATE) + r')\b',
    re.IGNORECASE,
)


class EmergencyDetector:
    """
    Hoku Health Care Emergency Detection System.

    Two-tier detection:
    - Tier 1: Sub-50ms regex keyword matching (synchronous, no LLM).
    - Tier 2: Fast Groq LLM fallback for ambiguous cases (async, 0.3s timeout).

    All methods are classmethods or staticmethods for stateless operation.
    """

    # Urgency level constants
    URGENCY_HIGH = "high"
    URGENCY_MODERATE = "moderate"
    URGENCY_NONE = "none"

    # Pre-formatted urgency responses
    URGENCY_RESPONSES: Dict[str, Dict[str, Any]] = {
        URGENCY_HIGH: {
            "reply": (
                "🚨 EMERGENCY DETECTED — HIGH URGENCY 🚨\n\n"
                "This appears to be a potentially life-threatening situation. "
                "Please call your local emergency number immediately:\n\n"
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
                "⚠️ URGENT SYMPTOMS DETECTED ⚠️\n\n"
                "Your symptoms suggest you should seek medical attention soon. "
                "Please contact a healthcare provider or visit an urgent care "
                "facility as soon as possible.\n\n"
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
        Tier 1: Detect potentially life-threatening symptoms via regex.

        Uses pre-compiled regex for sub-50ms synchronous matching.
        No LLM call — pure Python string matching.

        Args:
            message: User message (may be raw or HTML-escaped).

        Returns:
            Tuple[bool, str, str]: (is_emergency, urgency_level, reason)
            - is_emergency: True if emergency keywords detected.
            - urgency_level: "high", "moderate", or "none".
            - reason: The matched keyword(s) that triggered detection.

        Performance:
            - Typical execution: < 1ms on modern hardware
            - Worst case: < 10ms for 1000-character messages
            - Well under the 50ms requirement.
        """
        start_time = time.perf_counter()

        if not message or not isinstance(message, str):
            return (False, cls.URGENCY_NONE, "")

        # Check high urgency first (safety priority)
        high_match = _HIGH_URGENCY_PATTERN.search(message)
        if high_match:
            matched_keyword = high_match.group(0)
            elapsed = time.perf_counter() - start_time
            logger.critical(
                "EMERGENCY DETECTED (HIGH) in %.4fs: matched='%s', preview=%s...",
                elapsed,
                matched_keyword,
                message[:100],
            )
            return (True, cls.URGENCY_HIGH, matched_keyword)

        # Check moderate urgency
        moderate_match = _MODERATE_URGENCY_PATTERN.search(message)
        if moderate_match:
            matched_keyword = moderate_match.group(0)
            elapsed = time.perf_counter() - start_time
            logger.warning(
                "EMERGENCY DETECTED (MODERATE) in %.4fs: matched='%s', preview=%s...",
                elapsed,
                matched_keyword,
                message[:100],
            )
            return (True, cls.URGENCY_MODERATE, matched_keyword)

        elapsed = time.perf_counter() - start_time
        logger.debug("No emergency detected (Tier 1, %.4fs)", elapsed)
        return (False, cls.URGENCY_NONE, "")

    @classmethod
    async def detect_emergency_async(
        cls,
        message: str,
    ) -> Tuple[bool, str, str]:
        """
        Full detection pipeline: Tier 1 regex + optional Tier 2 LLM fallback.

        Steps:
        1. Run Tier 1 regex detection (fast, synchronous).
        2. If Tier 1 is negative AND message is ambiguous (long, narrative),
           run Tier 2 fast Groq check with 0.3s timeout.
        3. Return combined result.

        Args:
            message: User message to analyze.

        Returns:
            Tuple[bool, str, str]: (is_emergency, urgency_level, reason)
        """
        # Step 1: Tier 1 (always runs)
        is_emergency, urgency, reason = cls.detect_emergency(message)
        if is_emergency:
            return (is_emergency, urgency, reason)

        # Step 2: Tier 2 — ambiguous messages may need LLM fallback even if short.
        if cls._is_ambiguous(message):
            try:
                tier2_result = await asyncio.wait_for(
                    cls._tier2_llm_check(message),
                    timeout=ai_settings.EMERGENCY_CHECK_TIMEOUT,
                )
                if tier2_result[0]:  # is_emergency
                    logger.critical(
                        "EMERGENCY DETECTED (Tier 2 LLM): urgency=%s, reason=%s",
                        tier2_result[1],
                        tier2_result[2],
                    )
                    return tier2_result
            except asyncio.TimeoutError:
                logger.warning(
                    "Tier 2 emergency check timed out after %.3fs, using Tier 1 result",
                    ai_settings.EMERGENCY_CHECK_TIMEOUT,
                )
            except Exception as exc:
                logger.warning("Tier 2 emergency check failed: %s", exc)

        return (False, cls.URGENCY_NONE, "")

    @classmethod
    def _is_ambiguous(cls, message: str) -> bool:
        """
        Heuristic to determine if a message is ambiguous enough
        to warrant a Tier 2 LLM check.

        Ambiguous = long narrative without clear symptom keywords
        but containing health-related uncertainty language.
        """
        uncertainty_keywords = [
            "worried", "scared", "afraid", "don't know", "not sure",
            "something wrong", "feels wrong", "strange", "unusual",
            "never felt", "different", "weird", "alarming",
        ]
        uncertainty_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in uncertainty_keywords) + r')\b',
            re.IGNORECASE,
        )
        return bool(uncertainty_pattern.search(message))

    @classmethod
    async def _tier2_llm_check(cls, message: str) -> Tuple[bool, str, str]:
        """
        Tier 2: Fast Groq LLM check for ambiguous emergency cases.

        Uses llama-3.1-8b-instant with a strict 0.3s timeout.
        Only called when Tier 1 is negative but message is ambiguous.

        Args:
            message: The ambiguous user message.

        Returns:
            Tuple[bool, str, str]: (is_emergency, urgency_level, reason)
        """
        try:
            chain = cls._build_tier2_chain()
            if chain is None:
                raise RuntimeError("Tier 2 LLM chain unavailable")

            prompt_text = (
                "You are an emergency triage classifier. Analyze this patient message "
                "and determine if it describes a medical emergency requiring immediate "
                "professional attention.\n\n"
                "Respond ONLY with a JSON object in this exact format:\n"
                '{"is_emergency": true|false, "urgency": "high|moderate|none", "reason": "brief explanation"}\n\n'
                f'Patient message: "{message}"\n\n'
                "JSON response:"
            )

            result = await asyncio.to_thread(chain.invoke, {"text": prompt_text})

            raw_text = ""
            if isinstance(result, str):
                raw_text = result
            elif isinstance(result, dict):
                for key in ("text", "output", "content", "response"):
                    if key in result and isinstance(result[key], str):
                        raw_text = result[key]
                        break

            # Parse JSON
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
                cleaned = re.sub(r'\s*```$', '', cleaned)

            data = json.loads(cleaned)
            if isinstance(data, dict):
                is_emergency = bool(data.get("is_emergency", False))
                urgency = data.get("urgency", cls.URGENCY_NONE)
                reason = data.get("reason", "tier2_llm_ambiguous")

                # Validate urgency
                if urgency not in (cls.URGENCY_HIGH, cls.URGENCY_MODERATE, cls.URGENCY_NONE):
                    urgency = cls.URGENCY_HIGH if is_emergency else cls.URGENCY_NONE

                return (is_emergency, urgency, reason)

        except Exception as exc:
            logger.warning("Tier 2 LLM emergency check failed: %s", exc)

        return (False, cls.URGENCY_NONE, "")

    @classmethod
    def _build_tier2_chain(cls):
        """
        Build the Tier 2 emergency detection LLM chain.

        This helper exists so tests can patch the chain creation and
        the production code can build a valid LangChain LLMChain.
        """
        try:
            from langchain.chains import LLMChain
            from langchain_core.prompts import PromptTemplate
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=ai_settings.INTENT_MODEL,
                api_key=ai_settings.groq_api_key,
                temperature=0.0,
                max_tokens=64,
                request_timeout=ai_settings.EMERGENCY_CHECK_TIMEOUT,
            )
            prompt = PromptTemplate(template="{text}", input_variables=["text"])
            return LLMChain(llm=llm, prompt=prompt, verbose=False)
        except Exception as exc:
            logger.warning("Unable to build Tier 2 emergency chain: %s", exc)
            return None

    @classmethod
    def get_urgency_response(cls, urgency_level: str) -> Dict[str, Any]:
        """
        Return the pre-formatted urgent response for the detected urgency level.

        Args:
            urgency_level: "high" or "moderate".

        Returns:
            Dict with reply, suggestedSpecialist, severity, shouldSeeDoctor,
            intent, and confidence fields.
        """
        if urgency_level == cls.URGENCY_HIGH:
            return dict(cls.URGENCY_RESPONSES[cls.URGENCY_HIGH])
        elif urgency_level == cls.URGENCY_MODERATE:
            return dict(cls.URGENCY_RESPONSES[cls.URGENCY_MODERATE])
        else:
            # Unknown urgency defaults to high (safety bias)
            logger.warning("Unknown urgency_level='%s', defaulting to high", urgency_level)
            return dict(cls.URGENCY_RESPONSES[cls.URGENCY_HIGH])


# ------------------------------------------------------------------
# Backwards-compatible module-level functions (Day 0-6 API preserved)
# ------------------------------------------------------------------

def detect_emergency(message: str) -> bool:
    """
    Backwards-compatible wrapper: returns True/False for emergency detection.

    Preserves the Day 0-6 function signature. New code should use
    EmergencyDetector.detect_emergency() for full urgency metadata.

    Args:
        message: User message.

    Returns:
        bool: True if any emergency (high or moderate) detected.
    """
    is_emergency, _, _ = EmergencyDetector.detect_emergency(message)
    return is_emergency


def get_emergency_response() -> Dict[str, Any]:
    """
    Backwards-compatible wrapper: returns high-urgency emergency response.

    Preserves the Day 0-6 function signature. New code should use
    EmergencyDetector.get_urgency_response(urgency_level) for tiered responses.

    Returns:
        Dict with emergency response fields.
    """
    return EmergencyDetector.get_urgency_response(EmergencyDetector.URGENCY_HIGH)