"""
Hoku Health Care - Symptom Extractor (Day 6).

Dual-path symptom extraction:
1. FAST PATH: Regex keyword matching against SPECIALIST_MAP keys (< 10ms, no LLM).
2. LLM PATH (Fallback): Groq call with llama-3.1-8b-instant for complex text.
3. LATENCY GUARDRAIL: 0.2s timeout. On failure/timeout, defaults to ["fever"]
   (General Physician) to safeguard NFR-02 < 4s total latency.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from app.ai.config import ai_settings
from app.ai.specialist_mapper import SpecialistMapper

logger = logging.getLogger(__name__)

# Pre-compile regex for all known symptom keywords for O(n) matching
_SYMPTOM_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in SpecialistMapper.SPECIALIST_MAP.keys()) + r')\b',
    re.IGNORECASE,
)


def _normalize_symptoms(raw: List[str]) -> List[str]:
    """Deduplicate, lowercase, and strip symptom strings."""
    seen = set()
    result = []
    for s in raw:
        cleaned = s.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _regex_extract_symptoms(text: str) -> List[str]:
    """Fast regex extraction of known symptom keywords from text."""
    if not text or not isinstance(text, str):
        return []

    matches = _SYMPTOM_PATTERN.findall(text)
    normalized = _normalize_symptoms(matches)
    logger.debug("Regex extracted %d symptoms from text: %s", len(normalized), normalized)
    return normalized


def _is_complex_text(text: str) -> bool:
    """
    Heuristic: determine if text is complex enough to warrant LLM extraction.

    Complex = long sentences, no obvious symptom keywords, or narrative style.
    """
    if len(text) > 200:
        return True
    # If regex found nothing, text might have implied symptoms
    if not _regex_extract_symptoms(text):
        return True
    return False


async def _llm_extract_symptoms(text: str) -> List[str]:
    """
    Fallback LLM-based symptom extraction using Groq.

    Returns JSON {"symptoms": [...]} or empty list on failure.
    """
    try:
        from langchain.chains import LLMChain
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=ai_settings.SYMPTOM_EXTRACTION_MODEL,
            api_key=ai_settings.groq_api_key,
            temperature=0.0,
            max_tokens=128,
            request_timeout=ai_settings.SYMPTOM_EXTRACTION_TIMEOUT,
        )

        prompt_text = (
            "Extract medical symptoms from the following patient message. "
            "Return ONLY a JSON object with this exact format: "
            '{"symptoms": ["symptom1", "symptom2"]}. '
            "If no symptoms are mentioned, return {\"symptoms\": []}.\n\n"
            f'Patient message: "{text}"\n\n'
            "JSON response:"
        )

        chain = LLMChain(llm=llm, prompt=None, verbose=False)
        # Build a simple prompt invocation
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
        if isinstance(data, dict) and "symptoms" in data:
            symptoms = data["symptoms"]
            if isinstance(symptoms, list):
                normalized = _normalize_symptoms(symptoms)
                logger.info("LLM extracted symptoms: %s", normalized)
                return normalized

    except asyncio.TimeoutError:
        logger.warning("LLM symptom extraction timed out")
    except Exception as exc:
        logger.warning("LLM symptom extraction failed: %s", exc)

    return []


async def extract_symptoms_from_text(text: str) -> List[str]:
    """
    Extract symptoms from patient text using fast regex or LLM fallback.

    Strategy:
    1. Try regex keyword matching first (< 10ms).
    2. If regex is empty AND text is complex, try LLM with 0.2s timeout.
    3. If LLM fails/times out, default to ["fever"] -> General Physician.
    4. Normalize output: lowercase, stripped, deduplicated.

    Args:
        text: Raw or sanitized patient message.

    Returns:
        List[str]: Extracted symptom keywords (never empty on fallback).
    """
    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # FAST PATH: Regex keyword matching
    # ------------------------------------------------------------------
    regex_symptoms = _regex_extract_symptoms(text)
    if regex_symptoms:
        elapsed = time.perf_counter() - start_time
        logger.info("Fast-path symptom extraction: %s (%.3fms)", regex_symptoms, elapsed * 1000)
        return regex_symptoms

    # ------------------------------------------------------------------
    # LLM FALLBACK: Only for complex text without obvious keywords
    # ------------------------------------------------------------------
    if _is_complex_text(text):
        logger.info("No regex matches, attempting LLM symptom extraction for complex text")
        try:
            llm_task = asyncio.create_task(_llm_extract_symptoms(text))
            llm_symptoms = await asyncio.wait_for(
                llm_task,
                timeout=ai_settings.SYMPTOM_EXTRACTION_TIMEOUT,
            )
            if llm_symptoms:
                elapsed = time.perf_counter() - start_time
                logger.info("LLM symptom extraction: %s (%.3fms)", llm_symptoms, elapsed * 1000)
                return llm_symptoms
        except asyncio.TimeoutError:
            logger.warning(
                "Symptom extraction timed out after %.3fs, using default fallback",
                ai_settings.SYMPTOM_EXTRACTION_TIMEOUT,
            )
        except Exception as exc:
            logger.warning("Symptom extraction error: %s", exc)

    # ------------------------------------------------------------------
    # SAFETY DEFAULT: Never return empty — map to General Physician
    # ------------------------------------------------------------------
    logger.info("No symptoms extracted, defaulting to ['fever'] -> General Physician")
    return ["fever"]