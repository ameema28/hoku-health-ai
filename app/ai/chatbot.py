"""
Hoku Health Care - AI Chatbot Engine (Day 5: RAG pipeline integrated).

Core chatbot logic using Groq LLMs via LangChain 0.2.6 with per-user
conversation memory, intent classification, emergency detection, and
(Day 5) FAQ retrieval-augmented generation via pgvector.

Flow:
1. Emergency detection (regex, <50ms) -- bypasses LLM/RAG if emergency
2. Intent classification (llama-3.1-8b-instant, <500ms)
3. RAG lookup (Day 5): for GENERAL/SYMPTOM intents only, similarity_search
   against the Hoku FAQ collection. BOOKING/MEDICATION/EMERGENCY skip RAG
   entirely -- those flows are already well-served by the existing
   intent-aware prompt augmentation and don't benefit from FAQ grounding.
4. Intent-aware system prompt augmentation
5. Main LLM response generation (llama-3.3-70b-versatile), using
   RAG_SYSTEM_PROMPT when FAQ context was found, else the default
   SYSTEM_PROMPT
6. Persistence with intent metadata
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

from sqlalchemy.orm import Session

# LangChain imports at module level (required for test patching)
try:
    from langchain.chains import LLMChain
    from langchain_groq import ChatGroq

    LANGCHAIN_AVAILABLE = True
except ImportError as _import_exc:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None  # type: ignore
    LLMChain = None  # type: ignore
    logging.getLogger(__name__).warning("LangChain/Groq not installed: %s", _import_exc)

from app.ai.config import ai_settings
from app.ai.emergency_detector import detect_emergency, get_emergency_response
from app.ai.intent_classifier import IntentClassifier, IntentEnum
from app.ai.memory import HokuConversationMemory
from app.ai.prompts import chat_prompt_template, rag_chat_prompt_template
from app.ai.rag import HokuRAG
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# Day 5: intents for which a Hoku FAQ lookup is worthwhile. Booking and
# medication flows are handled by dedicated intent-aware prompts and
# don't benefit from FAQ grounding; emergency always bypasses the LLM.
_RAG_ELIGIBLE_INTENTS = {IntentEnum.GENERAL, IntentEnum.SYMPTOM}


class HokuChatbot:
    """
    Hoku Health Care AI Chatbot.

    Uses Groq via LangChain LLMChain:
    - Fast model (llama-3.1-8b-instant): Intent classification (Day 4).
    - Main model (llama-3.3-70b-versatile): Patient-facing response.

    Day 5 addition:
    - Lazily-initialized HokuRAG instance for FAQ-grounded responses.
    """

    def __init__(self) -> None:
        """Initialize with lazy-loaded Groq clients, intent classifier, and RAG."""
        self.groq_api_key = ai_settings.groq_api_key
        self.fast_model = ai_settings.GROQ_FAST_MODEL
        self.main_model = ai_settings.GROQ_MAIN_MODEL
        self.temperature = ai_settings.TEMPERATURE
        self.max_tokens = ai_settings.MAX_TOKENS
        self.timeout = ai_settings.GROQ_TIMEOUT_SECONDS
        self.max_retries = ai_settings.MAX_RETRIES

        self._fast_llm: Optional[Any] = None
        self._main_llm: Optional[Any] = None
        self._chain: Optional[Any] = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._rag: Optional[HokuRAG] = None  # Day 5: lazy, like the LLMs

    @property
    def fast_llm(self) -> Any:
        """Lazy initializer for fast classification model."""
        if self._fast_llm is None:
            if not LANGCHAIN_AVAILABLE or ChatGroq is None:
                return None
            try:
                self._fast_llm = ChatGroq(
                    model=self.fast_model,
                    api_key=self.groq_api_key,
                    temperature=0.0,
                    max_tokens=256,
                    request_timeout=self.timeout,
                )
                logger.info("Fast LLM (%s) initialized", self.fast_model)
            except Exception as exc:
                logger.warning("Failed to initialize fast LLM: %s", exc)
                self._fast_llm = None
        return self._fast_llm

    @property
    def main_llm(self) -> Any:
        """Lazy initializer for main response model."""
        if self._main_llm is None:
            if not LANGCHAIN_AVAILABLE or ChatGroq is None:
                logger.error("Main LLM unavailable: LangChain/Groq not installed")
                return None
            if not self.groq_api_key:
                logger.error("Main LLM unavailable: GROQ_API_KEY is empty")
                return None
            try:
                self._main_llm = ChatGroq(
                    model=self.main_model,
                    api_key=self.groq_api_key,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    request_timeout=self.timeout,
                )
                logger.info("Main LLM (%s) initialized", self.main_model)
            except Exception as exc:
                logger.warning("Failed to initialize main LLM: %s", exc)
                self._main_llm = None
        return self._main_llm

    @property
    def intent_classifier(self) -> IntentClassifier:
        """Lazy initializer for intent classifier."""
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifier()
        return self._intent_classifier

    @property
    def rag(self) -> HokuRAG:
        """Lazy initializer for the Day 5 RAG pipeline."""
        if self._rag is None:
            self._rag = HokuRAG()
        return self._rag

    def build_chain(self) -> Any:
        """Build LLMChain with the default system prompt and main model."""
        if self._chain is None:
            if LLMChain is None:
                raise RuntimeError("LLMChain not available")
            self._chain = LLMChain(llm=self.main_llm, prompt=chat_prompt_template, verbose=False)
            logger.info("LLMChain built with model=%s", self.main_model)
        return self._chain

    def _build_intent_context(self, intent: IntentEnum) -> str:
        """Build intent-aware context augmentation for the system prompt."""
        context_map = {
            IntentEnum.SYMPTOM: (
                "You are providing general symptom information only. "
                "Do not diagnose. Ask clarifying questions if needed. "
                "Assess severity and suggest appropriate specialists."
            ),
            IntentEnum.BOOKING: (
                "Guide the user to book an appointment via the patient dashboard. "
                "Explain the booking process and available specialties. "
                "Do not make appointments directly -- direct them to the portal."
            ),
            IntentEnum.MEDICATION: (
                "Remind the user to follow their doctor's prescription exactly. "
                "Provide general medication information only. "
                "Never suggest changing dosage or stopping medication without "
                "consulting a doctor."
            ),
            IntentEnum.GENERAL: (
                "Provide helpful, accurate health information. "
                "Be clear about what Hoku Health Care offers."
            ),
            IntentEnum.EMERGENCY: (
                "URGENT: This message indicates a potential emergency. "
                "Emphasize immediate medical attention. "
                "Provide emergency contact numbers for their region."
            ),
        }
        return context_map.get(intent, context_map[IntentEnum.GENERAL])

    async def get_response(
        self,
        message: str,
        user_id: int,
        db: Session,
        raw_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate chatbot response with intent classification, emergency
        detection, RAG retrieval, timeout, fallback, and memory.

        Steps:
        1. Emergency detection on RAW message (regex, <50ms) -- bypass everything
        2. Intent classification on sanitized message (async, <500ms)
        3. RAG lookup for GENERAL/SYMPTOM intents (Day 5)
        4. Build intent-aware context
        5. Load conversation memory from DB
        6. Generate response with main LLM (RAG-aware prompt if FAQ context found)
        7. Return with intent and confidence metadata

        Args:
            message: Sanitized user message (for LLM processing).
            user_id: Authenticated user ID.
            db: SQLAlchemy database session for memory loading.
            raw_message: Optional raw user message (before sanitization),
                used for emergency detection to avoid HTML-escaping issues.

        Returns:
            Dict with reply, suggestedSpecialist, severity, shouldSeeDoctor,
            intent, and confidence.
        """
        start_time = time.perf_counter()
        logger.info("Processing chat for user_id=%s", user_id)

        message_for_emergency = raw_message if raw_message is not None else message

        # ---------------------------------------------------------------
        # Step 1: Emergency Detection (SAFETY CRITICAL -- runs FIRST)
        # ---------------------------------------------------------------
        if detect_emergency(message_for_emergency):
            logger.critical("Emergency detected for user_id=%s -- bypassing LLM/RAG", user_id)
            emergency_response = get_emergency_response()
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Emergency response returned in %.3fs for user_id=%s", elapsed, user_id
            )
            return emergency_response

        # ---------------------------------------------------------------
        # Step 2: Intent Classification (async, <500ms)
        # ---------------------------------------------------------------
        intent_start = time.perf_counter()
        intent, confidence = await self.intent_classifier.classify_intent(message)
        intent_elapsed = time.perf_counter() - intent_start
        logger.info(
            "Intent=%s, confidence=%.2f, elapsed=%.3fs for user_id=%s",
            intent.value,
            confidence,
            intent_elapsed,
            user_id,
        )

        # ---------------------------------------------------------------
        # Step 3: RAG lookup (Day 5) -- GENERAL/SYMPTOM only
        # ---------------------------------------------------------------
        rag_start = time.perf_counter()
        faq_context = ""
        if intent in _RAG_ELIGIBLE_INTENTS:
            try:
                # Bound RAG lookup so DB similarity_search cannot exceed
                # the allotted latency budget. If it times out, proceed
                # without RAG context (fallback to general prompt).
                rag_task = asyncio.to_thread(self.rag.build_context, message)
                faq_context = await asyncio.wait_for(
                    rag_task, timeout=ai_settings.RAG_LOOKUP_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "RAG lookup timed out after %.3fs, skipping RAG",
                    ai_settings.RAG_LOOKUP_TIMEOUT,
                )
                faq_context = ""
            except Exception as exc:
                logger.warning(
                    "RAG lookup failed, falling back to general knowledge: %s", exc
                )
                faq_context = ""
        else:
            logger.debug("Skipping RAG for intent=%s (not RAG-eligible)", intent.value)
        rag_elapsed = time.perf_counter() - rag_start

        use_rag = bool(faq_context)
        if use_rag:
            logger.info(
                "RAG context found for user_id=%s (intent=%s, %.3fs)",
                user_id,
                intent.value,
                rag_elapsed,
            )
        else:
            logger.info(
                "No RAG context for user_id=%s (intent=%s, %.3fs) -- using default prompt",
                user_id,
                intent.value,
                rag_elapsed,
            )

        # ---------------------------------------------------------------
        # Step 4: Build intent-aware context
        # ---------------------------------------------------------------
        intent_context = self._build_intent_context(intent)
        # TODO Day 6: Route booking intents to appointment API
        # if intent == IntentEnum.BOOKING:
        #     pass

        # ---------------------------------------------------------------
        # Step 5: Check main LLM availability
        # ---------------------------------------------------------------
        if self.main_llm is None:
            logger.error("Main LLM unavailable -- returning fallback")
            fallback = self._fallback_response("LLM initialization failed")
            fallback["intent"] = intent.value
            fallback["confidence"] = confidence
            return fallback

        try:
            # -------------------------------------------------------
            # Step 6: Load conversation memory from database
            # -------------------------------------------------------
            memory_start = time.perf_counter()
            memory_manager = HokuConversationMemory(
                message_limit=ai_settings.MEMORY_MESSAGE_LIMIT,
                max_history_tokens=ai_settings.MEMORY_MAX_TOKENS,
            )
            memory = memory_manager.load_memory(user_id=user_id, db=db)
            memory_elapsed = time.perf_counter() - memory_start
            memory_vars = memory.load_memory_variables({"message": message})
            history_list = memory_vars.get("history", [])
            logger.info(
                "Memory loaded for user_id=%s in %.3fs (%d messages)",
                user_id,
                memory_elapsed,
                len(history_list),
            )

            # Adjust LLM timeout: intent + rag + memory + LLM must stay under 3.5s
            remaining_timeout = max(
                0.1,
                self.timeout - intent_elapsed - rag_elapsed - memory_elapsed,
            )

            # -------------------------------------------------------
            # Step 7: Build chain (RAG-aware if context found) and generate
            # -------------------------------------------------------
            if LLMChain is None:
                raise RuntimeError("LLMChain not available")

            prompt_template = rag_chat_prompt_template if use_rag else chat_prompt_template
            chain = LLMChain(llm=self.main_llm, prompt=prompt_template, memory=memory, verbose=False)

            invoke_input: Dict[str, str] = {"message": message, "context": intent_context}
            if use_rag:
                invoke_input["faq_context"] = faq_context

            llm_task = asyncio.to_thread(chain.invoke, invoke_input)
            result = await asyncio.wait_for(llm_task, timeout=remaining_timeout)

            elapsed = time.perf_counter() - start_time
            logger.info("Groq response in %.3fs for user_id=%s", elapsed, user_id)

            raw_text = self._extract_text_from_result(result)
            estimated_tokens = len(raw_text) // 4
            logger.info("Estimated response tokens for user_id=%s: ~%d", user_id, estimated_tokens)

            parsed = self._parse_llm_output(raw_text)
            reply = parsed.get("reply", "")
            if SAFETY_DISCLAIMER not in reply:
                reply = f"{reply} {SAFETY_DISCLAIMER}"
            parsed["reply"] = reply

            response = {
                "reply": parsed.get("reply", self._fallback_response()["reply"]),
                "suggestedSpecialist": parsed.get("suggestedSpecialist"),
                "severity": parsed.get("severity", "unknown"),
                "shouldSeeDoctor": parsed.get("shouldSeeDoctor", True),
                "intent": intent.value,
                "confidence": confidence,
            }
            return response

        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.warning("Groq timeout after %.3fs for user_id=%s", elapsed, user_id)
            fallback = self._fallback_response("timeout")
            fallback["intent"] = intent.value
            fallback["confidence"] = confidence
            return fallback

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception("Groq error after %.3fs for user_id=%s: %s", elapsed, user_id, exc)
            fallback = self._fallback_response(str(exc))
            fallback["intent"] = intent.value
            fallback["confidence"] = confidence
            return fallback

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

    def _parse_llm_output(self, text: str) -> Dict[str, Any]:
        """Parse JSON with multiple fallback strategies."""
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and "reply" in data:
                return data
        except json.JSONDecodeError:
            pass

        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "reply" in data:
                    return data
        except (json.JSONDecodeError, AttributeError):
            pass

        parsed: Dict[str, Any] = {
            "reply": text.strip(),
            "suggestedSpecialist": None,
            "severity": "unknown",
            "shouldSeeDoctor": True,
        }

        spec_match = re.search(r'"suggestedSpecialist"\s*:\s*"([^"]+)"', text)
        if spec_match:
            val = spec_match.group(1).strip()
            if val.lower() != "null":
                parsed["suggestedSpecialist"] = val

        sev_match = re.search(r'"severity"\s*:\s*"([^"]+)"', text)
        if sev_match:
            parsed["severity"] = sev_match.group(1)

        doc_match = re.search(r'"shouldSeeDoctor"\s*:\s*(true|false)', text)
        if doc_match:
            parsed["shouldSeeDoctor"] = doc_match.group(1).lower() == "true"

        return parsed

    def _fallback_response(self, reason: str = "unknown") -> Dict[str, Any]:
        """Safe fallback when LLM fails."""
        logger.info("Fallback response (reason=%s)", reason)
        return {
            "reply": (
                "I'm sorry, I couldn't process your request right now. "
                f"{SAFETY_DISCLAIMER}"
            ),
            "suggestedSpecialist": None,
            "severity": "unknown",
            "shouldSeeDoctor": True,
            "intent": "general",
            "confidence": 0.0,
        }
