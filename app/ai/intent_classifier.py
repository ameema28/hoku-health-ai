"""
Hoku Health Care - Intent Classification Module (Day 4).

Classifies user messages into 5 categories using a fast, lightweight LLM:
- symptom: User describes physical symptoms or health concerns
- booking: User wants to schedule an appointment
- medication: User asks about medicines, dosages, reminders
- general: General health information, wellness, platform questions
- emergency: Life-threatening symptoms requiring immediate attention

Uses llama-3.1-8b-instant for intent classification because:
- Cost: ~10x cheaper than llama-3.3-70b-versatile
- Latency: ~3x faster (critical for <500ms classification budget)
- Sufficient: Classification is a simpler task than generating clinical
  responses; 8B parameters handle 5-way classification reliably.

The main model (llama-3.3-70b-versatile) remains reserved for generating
empathetic, high-quality patient-facing responses where nuance matters.
"""

import asyncio
import json
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

# LangChain imports at module level (required for test patching)
try:
    from langchain.chains import LLMChain
    from langchain_groq import ChatGroq
    LANGCHAIN_AVAILABLE = True
except ImportError as _import_exc:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None  # type: ignore
    LLMChain = None  # type: ignore
    logging.getLogger(__name__).warning(
        "LangChain/Groq not installed: %s", _import_exc
    )

from app.ai.config import ai_settings
from app.ai.prompts import intent_classification_prompt_template

logger = logging.getLogger(__name__)


class IntentEnum(str, Enum):
    """
    Enumeration of supported intent categories.

    Using str Enum for clean JSON serialization and database storage.
    """
    SYMPTOM = "symptom"
    BOOKING = "booking"
    MEDICATION = "medication"
    GENERAL = "general"
    EMERGENCY = "emergency"


class IntentClassifier:
    """
    Hoku Health Care AI Intent Classifier.

    Wraps a fast Groq LLM (llama-3.1-8b-instant) for low-latency
    intent classification. Falls back to GENERAL on any failure.

    Design decisions:
    - Temperature 0.0: Intent classification must be deterministic.
      We want consistent categories for analytics and routing.
    - max_tokens 64: Classification output is tiny (JSON with 2 fields).
      Keeping this low reduces latency and cost.
    - request_timeout 0.5: Hard cutoff at 500ms to protect the 4s NFR.
      If classification times out, we fall back to GENERAL and proceed
      with the main LLM call rather than failing the entire request.
    """

    def __init__(self) -> None:
        """Initialize with lazy-loaded Groq client."""
        self.groq_api_key = ai_settings.groq_api_key
        self.model = ai_settings.INTENT_MODEL
        self.confidence_threshold = ai_settings.INTENT_CONFIDENCE_THRESHOLD
        self.timeout = ai_settings.INTENT_CLASSIFICATION_TIMEOUT
        self._llm: Optional[Any] = None
        self._chain: Optional[Any] = None

    @property
    def llm(self) -> Any:
        """Lazy initializer for the fast classification LLM."""
        if self._llm is None:
            if not LANGCHAIN_AVAILABLE or ChatGroq is None:
                logger.warning("Intent LLM unavailable: LangChain/Groq not installed")
                return None
            if not self.groq_api_key:
                logger.warning("Intent LLM unavailable: GROQ_API_KEY is empty")
                return None
            try:
                self._llm = ChatGroq(
                    model=self.model,
                    api_key=self.groq_api_key,
                    temperature=0.0,  # Deterministic for classification
                    max_tokens=64,    # Minimal: just JSON with intent + confidence
                    request_timeout=self.timeout,
                )
                logger.info("Intent classification LLM (%s) initialized", self.model)
            except Exception as exc:
                logger.warning("Failed to initialize intent LLM: %s", exc)
                self._llm = None
        return self._llm

    @property
    def chain(self) -> Any:
        """Build LLMChain with intent classification prompt."""
        if self._chain is None:
            if LLMChain is None:
                logger.error("LLMChain not available for intent classification")
                return None
            if self.llm is None:
                logger.error("Intent LLM not available, cannot build chain")
                return None
            self._chain = LLMChain(
                llm=self.llm,
                prompt=intent_classification_prompt_template,
                verbose=False,
            )
            logger.info("Intent classification chain built")
        return self._chain

    async def classify_intent(self, message: str) -> Tuple[IntentEnum, float]:
        """
        Classify the intent of a user message asynchronously.

        Steps:
        1. Run LLMChain in thread pool (asyncio.to_thread) to avoid
           blocking the event loop.
        2. Parse JSON output for intent label and confidence score.
        3. Validate against known IntentEnum values.
        4. If confidence < threshold, fall back to GENERAL.
        5. On any failure (timeout, parse error, LLM unavailable),
           return (GENERAL, 0.0) — never crash the chat flow.

        Args:
            message: Sanitized user message.

        Returns:
            Tuple[IntentEnum, float]: Classified intent and confidence
            score (0.0–1.0). Confidence is 0.0 on fallback.
        """
        start_time = time.perf_counter()

        # Fast path: if LLM unavailable, skip to fallback immediately
        if self.chain is None:
            logger.warning("Intent chain unavailable, falling back to GENERAL")
            return (IntentEnum.GENERAL, 0.0)

        try:
            # Run LLM call in thread pool to keep event loop free
            llm_task = asyncio.to_thread(
                self.chain.invoke,
                {"message": message}
            )
            result = await asyncio.wait_for(
                llm_task,
                timeout=self.timeout
            )

            elapsed = time.perf_counter() - start_time
            logger.info(
                "Intent classification completed in %.3fs for message_len=%d",
                elapsed,
                len(message),
            )

            # Extract text from LangChain result formats
            raw_text = self._extract_text_from_result(result)
            parsed = self._parse_intent_output(raw_text)

            intent_str = parsed.get("intent", "general")
            confidence = parsed.get("confidence", 0.0)

            # Validate intent string against known enum values
            try:
                intent = IntentEnum(intent_str.lower())
            except ValueError:
                logger.warning(
                    "Unknown intent '%s' from LLM, falling back to GENERAL",
                    intent_str,
                )
                intent = IntentEnum.GENERAL
                confidence = 0.0

            # Confidence threshold gate: low confidence → GENERAL
            if confidence < self.confidence_threshold:
                logger.info(
                    "Intent confidence %.2f below threshold %.2f, "
                    "falling back to GENERAL",
                    confidence,
                    self.confidence_threshold,
                )
                intent = IntentEnum.GENERAL
                confidence = 0.0

            logger.info(
                "Classified intent=%s, confidence=%.2f for message_len=%d",
                intent.value,
                confidence,
                len(message),
            )
            return (intent, confidence)

        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.warning(
                "Intent classification timeout after %.3fs (limit: %.3fs)",
                elapsed,
                self.timeout,
            )
            return (IntentEnum.GENERAL, 0.0)

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception(
                "Intent classification failed after %.3fs: %s",
                elapsed,
                exc,
            )
            return (IntentEnum.GENERAL, 0.0)

    @staticmethod
    def _extract_text_from_result(result: Any) -> str:
        """Extract text from LangChain return formats."""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            for key in ("text", "output", "content", "response"):
                if key in result and isinstance(result[key], str):
                    return result[key]
            return json.dumps(result)
        return str(result)

    def _parse_intent_output(self, text: str) -> Dict[str, Any]:
        """
        Parse intent classification JSON from LLM output.

        Expected format: {"intent": "symptom", "confidence": 0.92}

        Handles:
        - Pure JSON strings
        - Markdown code fences
        - Regex fallback for malformed JSON
        - Missing fields with safe defaults

        Args:
            text: Raw LLM response string.

        Returns:
            Dict with "intent" (str) and "confidence" (float).
        """
        if not text or not isinstance(text, str):
            return {"intent": "general", "confidence": 0.0}

        cleaned = text.strip()

        # Strip markdown fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'\s*```$', '', cleaned)

        # Try JSON parsing
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                intent = data.get("intent", "general")
                confidence = data.get("confidence", 0.0)
                # Normalize confidence to float
                if isinstance(confidence, str):
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.0
                elif not isinstance(confidence, (int, float)):
                    confidence = 0.0
                # Clamp confidence to [0.0, 1.0]
                confidence = max(0.0, min(1.0, float(confidence)))
                return {"intent": str(intent), "confidence": confidence}
        except (json.JSONDecodeError, ValueError):
            pass

        # Regex fallback: extract intent and confidence from malformed text
        intent_match = re.search(
            r'"intent"\s*:\s*"([^"]+)"',
            text,
            re.IGNORECASE,
        )
        confidence_match = re.search(
            r'"confidence"\s*:\s*([0-9.]+)',
            text,
            re.IGNORECASE,
        )

        fallback_intent = intent_match.group(1) if intent_match else "general"
        fallback_confidence = 0.0
        if confidence_match:
            try:
                fallback_confidence = float(confidence_match.group(1))
                fallback_confidence = max(0.0, min(1.0, fallback_confidence))
            except ValueError:
                pass

        return {
            "intent": fallback_intent,
            "confidence": fallback_confidence,
        }