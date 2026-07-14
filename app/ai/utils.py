"""
Hoku Health Care - AI Utility Functions.

Helper parsers for extracting structured metadata from LLM responses.
All functions are defensive: they never raise, always return safe defaults.
"""

import json
import logging
import re
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def parse_specialist_from_response(text: str) -> Optional[str]:
    """
    Extract the suggested medical specialist from LLM output.

    Tries JSON parsing first, then regex fallback. Returns None if
    the field is missing or unparseable.

    Args:
        text: Raw or parsed LLM response text.

    Returns:
        Optional[str]: Specialist name (e.g., "Cardiologist") or None.
    """
    # Try JSON extraction first
    try:
        data = extract_json_from_response(text)
        specialist = data.get("suggestedSpecialist")
        if specialist and isinstance(specialist, str) and specialist.lower() != "null":
            return specialist.strip()
    except Exception:
        pass

    # Regex fallback: look for "suggestedSpecialist" key
    match = re.search(r'"suggestedSpecialist"\s*:\s*"([^"]+)"', text)
    if match:
        val = match.group(1).strip()
        if val.lower() != "null":
            return val

    logger.debug("Could not parse specialist from response")
    return None


def parse_severity_from_response(text: str) -> str:
    """
    Extract severity level from LLM output.

    Args:
        text: Raw or parsed LLM response text.

    Returns:
        str: One of "mild", "moderate", "severe", or "unknown".
    """
    valid = {"mild", "moderate", "severe"}

    # JSON path
    try:
        data = extract_json_from_response(text)
        severity = data.get("severity", "unknown")
        if isinstance(severity, str) and severity.lower() in valid:
            return severity.lower()
    except Exception:
        pass

    # Regex fallback
    match = re.search(r'"severity"\s*:\s*"([^"]+)"', text)
    if match:
        val = match.group(1).lower().strip()
        if val in valid:
            return val

    logger.debug("Could not parse severity from response, defaulting to 'unknown'")
    return "unknown"


def parse_should_see_doctor(text: str) -> bool:
    """
    Extract the shouldSeeDoctor flag from LLM output.

    In clinical safety terms, we bias toward True (recommend doctor)
    when parsing is ambiguous — it's safer to over-refer than under-refer.

    Args:
        text: Raw or parsed LLM response text.

    Returns:
        bool: True if the user should see a doctor.
    """
    # JSON path
    try:
        data = extract_json_from_response(text)
        flag = data.get("shouldSeeDoctor")
        if isinstance(flag, bool):
            return flag
    except Exception:
        pass

    # Regex fallback
    match = re.search(r'"shouldSeeDoctor"\s*:\s*(true|false)', text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"

    # Safety bias: when in doubt, recommend professional care
    logger.debug("Ambiguous shouldSeeDoctor, defaulting to True (safety bias)")
    return True


def extract_json_from_response(text: str) -> Dict:
    """
    Extract a JSON object from LLM text output.

    Handles:
    - Pure JSON strings
    - Markdown code fences (```json ... ```)
    - Trailing/leading whitespace and newlines

    Args:
        text: Raw LLM response string.

    Returns:
        Dict: Parsed JSON object (empty dict on failure).

    Raises:
        No exceptions — returns empty dict on any failure.
    """
    if not text or not isinstance(text, str):
        return {}

    cleaned = text.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        # Remove opening fence and optional language tag
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        # Remove closing fence
        cleaned = re.sub(r'\s*```$', '', cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except json.JSONDecodeError:
        logger.debug("JSON decode failed for text length=%d", len(cleaned))
        return {}
    except Exception as exc:
        logger.warning("Unexpected error parsing JSON: %s", exc)
        return {}