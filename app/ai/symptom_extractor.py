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
from typing import List

from app.ai.config import ai_settings
from app.ai.specialist_mapper import SpecialistMapper

logger = logging.getLogger(__name__)

_SYMPTOM_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in SpecialistMapper.SPECIALIST_MAP.keys()) + r')\b',
    re.IGNORECASE,
)

_MEDICAL_HINTS_PATTERN = re.compile(
    r'\b(feel|feeling|hurt|pain|ache|sick|unwell|dizzy|nausea|tired|'
    r'fatigue|weak|fever|cough|rash|swelling|bruise|wound|burn|'
    r'fracture|sprain|bleed|vomit|diarrhea|headache|chest|stomach|'
    r'back|leg|arm|throat|ear|eye|heart|lung|symptom|discomfort)\b',
    re.IGNORECASE,
)


def _normalize_symptoms(raw: List[str]) -> List[str]:
    seen = set()
    result = []
    for s in raw:
        cleaned = s.strip().lower()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def _regex_extract_symptoms(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    matches = _SYMPTOM_PATTERN.findall(text)
    return _normalize_symptoms(matches)


def _is_complex_text(text: str) -> bool:
    if len(text) > 200:
        return True
    if not _regex_extract_symptoms(text):
        return bool(_MEDICAL_HINTS_PATTERN.search(text))
    return False


def _llm_extract_symptoms_sync(text: str) -> List[str]:
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGroq(
            model=ai_settings.SYMPTOM_EXTRACTION_MODEL,
            api_key=ai_settings.groq_api_key,
            temperature=0.0,
            max_tokens=128,
            request_timeout=ai_settings.SYMPTOM_EXTRACTION_TIMEOUT,
        )

        messages = [
            SystemMessage(content=(
                "You are a medical symptom extractor. Extract only the symptoms "
                "mentioned in the patient's message. Return ONLY a JSON object "
                'with this exact format: {"symptoms": ["symptom1", "symptom2"]}. '
                'If no symptoms are mentioned, return {"symptoms": []}.'
            )),
            HumanMessage(content=f'Patient message: "{text}"'),
        ]

        result = llm.invoke(messages)

        raw_text = ""
        if isinstance(result, str):
            raw_text = result
        elif hasattr(result, "content"):
            raw_text = result.content
        elif isinstance(result, dict):
            for key in ("text", "output", "content", "response"):
                if key in result and isinstance(result[key], str):
                    raw_text = result[key]
                    break

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*```$', '', cleaned)

        data = json.loads(cleaned)
        if isinstance(data, dict) and "symptoms" in data:
            symptoms = data["symptoms"]
            if isinstance(symptoms, list):
                return _normalize_symptoms(symptoms)

    except Exception as exc:
        logger.debug("LLM symptom extraction failed: %s", exc)

    return []


async def _llm_extract_symptoms(text: str) -> List[str]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_llm_extract_symptoms_sync, text),
            timeout=ai_settings.SYMPTOM_EXTRACTION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Symptom extraction timed out after %.3fs",
            ai_settings.SYMPTOM_EXTRACTION_TIMEOUT,
        )
        return []
    except Exception as exc:
        logger.warning("Symptom extraction error: %s", exc)
        return []


async def extract_symptoms_from_text(text: str) -> List[str]:
    start_time = time.perf_counter()

    regex_symptoms = _regex_extract_symptoms(text)
    if regex_symptoms:
        elapsed = time.perf_counter() - start_time
        logger.info("Fast-path symptom extraction: %s (%.3fms)", regex_symptoms, elapsed * 1000)
        return regex_symptoms

    if _is_complex_text(text):
        logger.info("No regex matches, attempting LLM symptom extraction for complex text")
        try:
            llm_symptoms = await _llm_extract_symptoms(text)
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

    logger.info("No symptoms extracted, defaulting to ['fever'] -> General Physician")
    return ["fever"]