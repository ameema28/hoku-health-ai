"""
Hoku Health Care - Safety Guardrails Module (Day 7).

Post-LLM safety layer that validates AI responses for clinical safety:
- No definitive diagnoses
- No prescription/dosage advice
- Mandatory disclaimer present
- Sanitization of prohibited content with 3-strike retry logic
"""

import logging
import re
from typing import List, Tuple

from app.ai.config import ai_settings
from app.crud.crud_safety import log_safety_violation
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# FORBIDDEN PATTERN DEFINITIONS
# ------------------------------------------------------------------
# These patterns detect language that violates clinical safety rules.
# Each pattern is conservative — we bias toward flagging over missing.
#
# Definitive diagnosis patterns: language that asserts a medical condition
_DIAGNOSIS_PATTERNS: List[re.Pattern] = [
    re.compile(r'\byou\s+have\s+(?:a\s+)?(?:\w+\b(?:itis|osis|emia|oma|pathy|syndrome)|pneumonia|bronchitis|hypertension|diabetes|asthma|infection|cold|flu|migraine|headache|cough|fever)', re.IGNORECASE),
    # ------------------------------------------------------------------
    # REMOVED Day 8.1 — catastrophic false-positive pattern:
    #     r'\byou\s+have\s+(?:a\s+)?[A-Za-z][A-Za-z\s\'-]{0,100}\b'
    #
    # It matched ANY occurrence of "you have" plus up to 100 following
    # characters, so ordinary phrasing was flagged as a definitive diagnosis:
    #     "If you have any questions, please reach out"      -> FLAGGED
    #     "Since you have been experiencing fatigue"          -> FLAGGED
    #     "You have several options for booking"              -> FLAGGED
    # while missing real clinical language that omits the pronoun:
    #     "Diabetes symptoms include thirst and fatigue"      -> not flagged
    #
    # Worse, sanitize_response() had NO counterpart replacement for it, so
    # the offending text survived sanitisation unchanged. validate ->
    # sanitize -> validate looped three times without converging and every
    # such reply was replaced by the 3-strike SAFETY_FALLBACK_RESPONSE.
    # Net effect: the guardrail silently destroyed correct, safe answers.
    #
    # The pattern immediately above already covers genuine diagnosis
    # assertions ("you have diabetes", "you have bronchitis", "...itis",
    # "...osis", etc.), so removing this line narrows false positives
    # without weakening clinical coverage.
    #
    # INVARIANT for anyone editing this list: every pattern here MUST have a
    # corresponding entry in sanitize_response's diagnosis_replacements.
    # A validate pattern with no sanitiser cannot converge and will always
    # burn all three strikes. tests/test_chatbot.py::TestSafetyConvergence
    # enforces this.
    # ------------------------------------------------------------------
    re.compile(r'\byou\s+are\s+suffering\s+from\b', re.IGNORECASE),
    re.compile(r'\bdiagnosis\s+is\s+', re.IGNORECASE),
    re.compile(r'\byour\s+condition\s+is\s+', re.IGNORECASE),
    re.compile(r'\byou\s+have\s+been\s+diagnosed\s+with\b', re.IGNORECASE),
    re.compile(r'\bthis\s+is\s+(?:a\s+)?(?:\w+\s+)?(?:infection|disease|disorder|condition)\b', re.IGNORECASE),
    re.compile(r'\byou\s+have\s+(?:developed|contracted)\s+\w+', re.IGNORECASE),
    re.compile(r'\bit\s+is\s+clear\s+that\s+you\s+have\b', re.IGNORECASE),
    re.compile(r'\byour\s+symptoms\s+indicate\s+', re.IGNORECASE),
    re.compile(r'\bthis\s+confirms\s+', re.IGNORECASE),
]

# Prescription/dosage patterns: language that gives specific medical instructions
_PRESCRIPTION_PATTERNS: List[re.Pattern] = [
    re.compile(r'\btake\s+\d+\s*(?:mg|g|mcg|ml|units?)\b', re.IGNORECASE),
    re.compile(r'\bprescribe\s+\w+', re.IGNORECASE),
    re.compile(r'\bdosage\s+(?:of\s+)?\d+(?:\s*(?:mg|g|mcg|ml|units?))?\b', re.IGNORECASE),
    re.compile(r'\btake\s+(?:[A-Za-z\'-]+\s+){1,4}(?:twice|three\s+times|once|daily|every\s+\d+\s+hours?)\b', re.IGNORECASE),
    re.compile(r'\bstart\s+taking\s+\w+', re.IGNORECASE),
    re.compile(r'\bstop\s+taking\s+\w+', re.IGNORECASE),
    re.compile(r'\bincrease\s+(?:your\s+)?(?:dose|dosage|medication)\b', re.IGNORECASE),
    re.compile(r'\bdecrease\s+(?:your\s+)?(?:dose|dosage|medication)\b', re.IGNORECASE),
    re.compile(r'\bapply\s+\d+\s*(?:mg|g|ml|drops?)\s+(?:of\s+)?\w+\b', re.IGNORECASE),
    re.compile(r'\binject\s+\d+\s*(?:mg|g|ml|units?)\b', re.IGNORECASE),
    re.compile(r'\buse\s+\d+\s*(?:mg|g|ml|drops?|puffs?)\b', re.IGNORECASE),
]

# Disclaimer pattern: check for the mandatory safety disclaimer
_DISCLAIMER_PATTERN = re.compile(
    re.escape(SAFETY_DISCLAIMER),
    re.IGNORECASE,
)


class SafetyGuardrails:
    """
    Clinical safety validator and sanitizer for AI responses.

    Validates that responses:
    1. Do not contain definitive diagnosis language
    2. Do not contain prescription/dosage advice
    3. Include the mandatory safety disclaimer

    Provides sanitization to strip or replace prohibited phrasing,
    and a 3-strike retry mechanism for enforcing safety compliance.
    """

    # Violation type constants (match SafetyLog.violation_type values)
    VIOLATION_DIAGNOSIS = "diagnosis_attempt"
    VIOLATION_PRESCRIPTION = "prescription_advice"
    VIOLATION_MISSING_DISCLAIMER = "missing_disclaimer"
    VIOLATION_SANITIZED = "safety_sanitized"

    @classmethod
    def validate_response(cls, text: str) -> Tuple[bool, List[str]]:
        """
        Validate an AI response against clinical safety rules.

        Args:
            text: The AI-generated response text to validate.

        Returns:
            Tuple[bool, List[str]]: (is_safe, list_of_violation_types)
            is_safe is True if no violations detected.
            violation_types contains one or more of:
                - "diagnosis_attempt"
                - "prescription_advice"
                - "missing_disclaimer"
        """
        if not text or not isinstance(text, str):
            logger.warning("validate_response called with empty or non-string input")
            return (False, [cls.VIOLATION_MISSING_DISCLAIMER])

        violations: List[str] = []

        # Check 1: Definitive diagnosis language
        diagnosis_hits = []
        for pattern in _DIAGNOSIS_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                diagnosis_hits.extend(matches)
        if diagnosis_hits:
            violations.append(cls.VIOLATION_DIAGNOSIS)
            logger.warning(
                "Diagnosis violation detected: %d matches",
                len(diagnosis_hits),
            )

        # Check 2: Prescription/dosage advice
        prescription_hits = []
        for pattern in _PRESCRIPTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                prescription_hits.extend(matches)
        if prescription_hits:
            violations.append(cls.VIOLATION_PRESCRIPTION)
            logger.warning(
                "Prescription violation detected: %d matches",
                len(prescription_hits),
            )

        # Check 3: Mandatory disclaimer
        if not _DISCLAIMER_PATTERN.search(text):
            violations.append(cls.VIOLATION_MISSING_DISCLAIMER)
            logger.warning("Missing disclaimer violation detected")

        is_safe = len(violations) == 0
        if is_safe:
            logger.debug("Response passed all safety checks")

        return (is_safe, violations)

    @classmethod
    def add_disclaimer(cls, text: str) -> str:
        """
        Append the mandatory clinical disclaimer if missing.

        Args:
            text: The AI response text.

        Returns:
            str: Text with disclaimer appended if it was missing.
        """
        if not text or not isinstance(text, str):
            return f"{ai_settings.SAFETY_FALLBACK_RESPONSE} {SAFETY_DISCLAIMER}"

        if SAFETY_DISCLAIMER not in text:
            text = f"{text} {SAFETY_DISCLAIMER}"
            logger.debug("Disclaimer appended to response")
        return text

    @classmethod
    def sanitize_response(cls, text: str) -> str:
        """
        Strip or replace prohibited diagnostic/prescription phrasing.

        Replaces diagnosis assertions with safe alternatives and removes
        specific dosage instructions. Always ensures disclaimer is present.

        Args:
            text: The AI response text to sanitize.

        Returns:
            str: Sanitized text safe for patient consumption.
        """
        if not text or not isinstance(text, str):
            return f"{ai_settings.SAFETY_FALLBACK_RESPONSE} {SAFETY_DISCLAIMER}"

        sanitized = text

        # Replace diagnosis language with safe alternatives
        diagnosis_replacements = [
            (r'\byou\s+have\s+(?:a\s+)?(\w+(?:itis|osis|emia|oma|pathy|syndrome)|pneumonia|bronchitis|hypertension|diabetes|asthma|infection|cold|flu|migraine)\b',
             r'You mentioned symptoms that could be related to \1, but only a doctor can confirm this.'),
            (r'\byou\s+are\s+suffering\s+from\b', 'You described symptoms that may relate to'),
            (r'\bdiagnosis\s+is\s+', 'a proper diagnosis can only be made by a doctor after examining you — possible considerations include '),
            (r'\byour\s+condition\s+is\s+', 'your symptoms could be related to several conditions, and '),
            (r'\byou\s+have\s+been\s+diagnosed\s+with\b', 'a diagnosis of'),
            (r'\bthis\s+is\s+(?:a\s+)?(\w+\s+(?:infection|disease|disorder|condition))\b',
             r'this could be related to \1, but'),
            (r'\byou\s+have\s+(?:developed|contracted)\s+(\w+)\b',
             r'you mentioned symptoms that might suggest \1, but'),
            (r'\bit\s+is\s+clear\s+that\s+you\s+have\b', 'your symptoms suggest you may have'),
            (r'\byour\s+symptoms\s+indicate\s+', 'your symptoms could be associated with'),
            (r'\bthis\s+confirms\s+', 'this suggests the possibility of'),
        ]

        for pattern, replacement in diagnosis_replacements:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # Replace prescription language with safe alternatives
        prescription_replacements = [
            (r'\btake\s+\d+\s*(?:mg|g|mcg|ml|units?)\s+(?:of\s+)?(\w+)\b',
             r'Please consult your doctor about the appropriate dosage of \1.'),
            (r'\bprescribe\s+(\w+)\b',
             r'Your doctor can determine if \1 is appropriate for you.'),
            (r'\bdosage\s+(?:of\s+)?\d+\b',
             'Your doctor will determine the correct dosage for you.'),
            (r'\btake\s+(?:\w+\s+)?(\w+)\s+(?:twice|three\s+times|once|daily|every\s+\d+\s+hours?)\b',
             r'Please follow your doctor\'s instructions for taking \1.'),
            (r'\bstart\s+taking\s+(\w+)\b',
             r'Discuss starting \1 with your doctor first.'),
            # Day 8.1: the old replacement was
            #     r'Never stop taking \1 without consulting your doctor.'
            # which REPRODUCES the trigger phrase "stop taking <drug>". The
            # pattern re-matched its own output on every pass, so the 3-strike
            # loop could never converge and any reply mentioning stopping a
            # medication was replaced by the safety fallback. The wording
            # below carries the same clinical meaning without the trigger.
            (r'\bstop\s+taking\s+(\w+)\b',
             r'please do not discontinue \1 on your own — speak with your doctor first'),
            (r'\bincrease\s+(?:your\s+)?(?:dose|dosage|medication)\b',
             'Any changes to your medication should be approved by your doctor.'),
            (r'\bdecrease\s+(?:your\s+)?(?:dose|dosage|medication)\b',
             'Any changes to your medication should be approved by your doctor.'),
            (r'\bapply\s+\d+\s*(?:mg|g|ml|drops?)\s+(?:of\s+)?(\w+)\b',
             r'Your doctor can advise on how to apply \1.'),
            (r'\binject\s+\d+\s*(?:mg|g|ml|units?)\b',
             'Injection administration should only be done under medical supervision.'),
            (r'\buse\s+\d+\s*(?:mg|g|ml|drops?|puffs?)\b',
             'Please confirm the correct usage with your doctor or pharmacist.'),
        ]

        for pattern, replacement in prescription_replacements:
            sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

        # Ensure disclaimer is present
        sanitized = cls.add_disclaimer(sanitized)

        logger.info("Response sanitized: %d diagnosis + %d prescription patterns processed",
                    len(diagnosis_replacements), len(prescription_replacements))
        return sanitized

    @classmethod
    def apply_3_strike_safety(
        cls,
        text: str,
        user_id: int,
        db=None,
    ) -> Tuple[str, List[str], str]:
        """
        Apply the 3-strike safety retry mechanism.

        Validates the response. If unsafe, sanitizes and re-validates.
        Repeats up to SAFETY_MAX_RETRIES times. If still unsafe after
        max retries, returns the hardcoded safe fallback.

        Args:
            text: The initial AI-generated response.
            user_id: The authenticated user's ID (for logging).
            db: Optional SQLAlchemy session for safety log persistence.

        Returns:
            Tuple[str, List[str], str]: (final_response, violations, severity)
            final_response: The safe response to return.
            violations: List of all violation types detected.
            severity: Overall severity ("high", "moderate", "low").
        """
        max_retries = ai_settings.SAFETY_MAX_RETRIES
        all_violations: List[str] = []
        current_text = text
        final_severity = "low"

        for attempt in range(1, max_retries + 1):
            is_safe, violations = cls.validate_response(current_text)
            all_violations.extend(v for v in violations if v not in all_violations)

            if is_safe:
                logger.info(
                    "Safety check passed on attempt %d/%d for user_id=%s",
                    attempt,
                    max_retries,
                    user_id,
                )
                return (current_text, all_violations, final_severity)

            logger.warning(
                "Safety violation on attempt %d/%d for user_id=%s: %s",
                attempt,
                max_retries,
                user_id,
                violations,
            )

            # Determine severity based on violation types
            if cls.VIOLATION_DIAGNOSIS in violations or cls.VIOLATION_PRESCRIPTION in violations:
                final_severity = "high"
            elif cls.VIOLATION_MISSING_DISCLAIMER in violations and final_severity == "low":
                final_severity = "moderate"

            # Log the violation if DB session available
            if db is not None:
                try:
                    log_safety_violation(
                        db=db,
                        user_id=user_id,
                        message="[post-llm safety check]",
                        ai_response=current_text[:1000],
                        violation_type=violations[0] if violations else cls.VIOLATION_SANITIZED,
                        severity=final_severity,
                    )
                except Exception as log_exc:
                    logger.warning("Failed to log safety violation: %s", log_exc)

            # Sanitize for next attempt
            current_text = cls.sanitize_response(current_text)

        # All retries exhausted — return hardcoded safe fallback
        logger.critical(
            "SAFETY 3-STRIKE FALLBACK triggered for user_id=%s after %d attempts",
            user_id,
            max_retries,
        )

        fallback = (
            f"{ai_settings.SAFETY_FALLBACK_RESPONSE} {SAFETY_DISCLAIMER}"
        )

        # Log the 3-strike fallback as a high-severity event
        if db is not None:
            try:
                log_safety_violation(
                    db=db,
                    user_id=user_id,
                    message="[3-strike safety fallback triggered]",
                    ai_response=fallback[:1000],
                    violation_type="safety_3_strike_fallback",
                    severity="high",
                )
            except Exception as log_exc:
                logger.warning("Failed to log 3-strike fallback: %s", log_exc)

        return (fallback, all_violations, "high")