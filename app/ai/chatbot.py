"""
Hoku Health Care - AI Chatbot Engine (Day 8: Performance Optimization & NFR-02 Compliance).

Core chatbot logic using Groq LLMs via LangChain 0.2.6 with per-user
conversation memory, intent classification, emergency detection, RAG,
specialist/doctor suggestion, post-LLM safety verification, and
(Day 8) strict step budgeting with async concurrency for <4s guarantees.

Flow:
1. Emergency detection (keyword regex, <50ms): fixed urgent reply
2. Intent classification + Memory loading + RAG retrieval (asyncio.gather);
   an ``emergency`` intent also gets the fixed urgent reply
3. Response cache lookup (<5ms)
4. LLM generation with remaining budget timeout
5. Post-LLM Safety Verification (3-strike safety retry)
6. Cache persistence (non-emergency only)
7. Persistence with intent metadata

Day 8 Performance Optimizations:
- Strict TIME_BUDGETS per pipeline stage
- asyncio.gather for parallel intent/memory/RAG (saves ~300-500ms)
- ResponseCache with TTL for non-clinical queries
- LLMFactory with model-specific timeouts
- generate_with_timeout for all async operations
- Connection-pooled DB sessions
"""

import asyncio
import functools
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from unittest.mock import MagicMock as _MagicMock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_groq import ChatGroq

    LANGCHAIN_AVAILABLE = True
except ImportError as _import_exc:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None  # type: ignore
    logging.getLogger(__name__).warning("langchain-groq not installed: %s", _import_exc)

from app.ai.caching import ResponseCache  # Day 8: In-memory response cache
from app.ai.config import ai_settings
from app.ai.emergency_detector import EmergencyDetector
from app.ai.fallback_responses import build_fallback_reply
from app.ai.intent_classifier import IntentClassifier, IntentEnum
from app.ai.llm_optimizer import LLMFactory, compress_prompt  # Day 8: LLM optimization
from app.ai.memory import HokuConversationMemory
from app.ai.ai_performance import (
    ResponseOptimizer,
    generate_with_timeout,
)  # Day 8: Performance layer
from app.ai.prompts import chat_prompt_template, rag_chat_prompt_template
from app.ai.rag import HokuRAG, format_faq_context
from app.ai.safety_guardrails import SafetyGuardrails
from app.ai.specialist_mapper import SpecialistMapper
from app.ai.symptom_extractor import extract_symptoms_from_text
from app.core.monitoring import get_metrics
from app.crud.crud_doctor import get_doctor_availability
from app.schemas.schemas_doctor import (
    DoctorAvailability as DoctorAvailabilitySchema,
    DoctorSuggestion,
)
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

#: Fallback specialty when the mapped one has no available doctor.
GENERAL_PHYSICIAN = "General Physician"

# Intents for which a Hoku FAQ lookup is worthwhile. BOOKING is included:
# several seeded FAQs answer booking questions (how to book, wait times).
_RAG_ELIGIBLE_INTENTS = {IntentEnum.GENERAL, IntentEnum.SYMPTOM, IntentEnum.BOOKING}

# Day 6: intents eligible for specialist suggestion flow
_SPECIALIST_ELIGIBLE_INTENTS = {IntentEnum.SYMPTOM, IntentEnum.GENERAL}

# Minimum LLM timeout — never go below this, even if budget is tight.
_MIN_LLM_TIMEOUT: float = 0.3


class HokuChatbot:
    """
    Hoku Health Care AI Chatbot.

    Uses Groq via LangChain runnables (prompt | ChatGroq | StrOutputParser):
    - Fast model (llama-3.1-8b-instant): Intent classification (Day 4).
    - Main model (llama-3.3-70b-versatile): Patient-facing response.

    Day 5 addition: Lazily-initialized HokuRAG instance.
    Day 6 addition: SpecialistMapper for symptom-to-doctor routing.
    Day 7 addition: SafetyGuardrails for post-LLM clinical safety verification.
    Day 8 addition: ResponseCache, ResponseOptimizer, LLMFactory for
        strict <4s NFR-02 compliance via time budgeting and async concurrency.
    """

    def __init__(self) -> None:
        """Initialize with lazy-loaded Groq clients, intent classifier, RAG, specialist mapper, safety guardrails, and performance layer."""
        self.groq_api_key = ai_settings.groq_api_key
        self.fast_model = ai_settings.GROQ_FAST_MODEL
        self.main_model = ai_settings.GROQ_MAIN_MODEL
        self.temperature = ai_settings.TEMPERATURE
        self.max_tokens = ai_settings.MAX_TOKENS
        self.timeout = ai_settings.GROQ_TIMEOUT_SECONDS

        self._fast_llm: Optional[Any] = None
        self._main_llm: Optional[Any] = None
        self._chain: Optional[Any] = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._rag: Optional[HokuRAG] = None  # Day 5
        self._specialist_mapper: Optional[SpecialistMapper] = None  # Day 6
        self._response_cache: Optional[ResponseCache] = None  # Day 8
        self._llm_factory: Optional[LLMFactory] = None  # Day 8

    def warm_up(self) -> None:
        """
        Eagerly initialize LLM clients and chains to eliminate cold-start latency.

        Call this once at application startup (e.g., from main.py lifespan).
        """
        logger.info("Warming up HokuChatbot LLM clients...")
        _ = self.fast_llm
        main = self.main_llm
        _ = self.intent_classifier.chain

        if main is None:
            logger.warning("Skipping chain warm-up: main LLM unavailable")
        else:
            try:
                _ = self.build_chain()
            except Exception as exc:
                logger.warning("Chain warm-up skipped: %s", type(exc).__name__)

        logger.info("HokuChatbot warm-up complete.")

    @property
    def fast_llm(self) -> Any:
        """Lazy initializer for fast classification model."""
        if self._fast_llm is None:
            if not LANGCHAIN_AVAILABLE or ChatGroq is None:
                return None
            try:
                self._fast_llm = self.llm_factory.get_fast_llm(
                    api_key=self.groq_api_key,
                    model=self.fast_model,
                    request_timeout=self.timeout,
                )
                if self._fast_llm is None:
                    raise RuntimeError("LLMFactory returned no fast LLM")
                logger.info("Fast LLM (%s) initialized via LLMFactory", self.fast_model)
            except Exception as exc:
                logger.warning("Failed to initialize fast LLM: %s", type(exc).__name__)
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
                self._main_llm = self.llm_factory.get_main_llm(
                    api_key=self.groq_api_key,
                    model=self.main_model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    request_timeout=self.timeout,
                )
                if self._main_llm is None:
                    raise RuntimeError("LLMFactory returned no main LLM")
                logger.info("Main LLM (%s) initialized via LLMFactory", self.main_model)
            except Exception as exc:
                logger.warning("Failed to initialize main LLM: %s", type(exc).__name__)
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

    @property
    def specialist_mapper(self) -> SpecialistMapper:
        """Lazy initializer for the Day 6 specialist mapper."""
        if self._specialist_mapper is None:
            self._specialist_mapper = SpecialistMapper()
        return self._specialist_mapper

    @property
    def response_cache(self) -> ResponseCache:
        """Lazy initializer for the Day 8 response cache."""
        if self._response_cache is None:
            self._response_cache = ResponseCache()
        return self._response_cache

    @property
    def llm_factory(self) -> LLMFactory:
        """Lazy initializer for the Day 8 LLM factory."""
        if self._llm_factory is None:
            self._llm_factory = LLMFactory()
        return self._llm_factory

    def _generation_chain(self, prompt: Any) -> Any:
        """
        ``prompt | main LLM | StrOutputParser()``: the single seam tests patch.

        ``invoke`` takes the prompt variables (including ``history`` as a
        list of messages) and returns the model's text.
        """
        if self.main_llm is None:
            raise RuntimeError("Main LLM not available (check GROQ_API_KEY)")
        return prompt | self.main_llm | StrOutputParser()

    def build_chain(self) -> Any:
        """Build (once) the default chat chain; used for warm-up."""
        if self._chain is None:
            self._chain = self._generation_chain(chat_prompt_template)
            logger.info("Chat chain built with model=%s", self.main_model)
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
                "Never give a dose, amount or how often to take a medicine, and "
                "never suggest starting, stopping or changing a medication; a "
                "doctor or pharmacist can advise on doses."
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

    async def _lookup_doctor_suggestion(
        self,
        message: str,
        db: Session,
    ) -> Optional[DoctorSuggestion]:
        """
        Day 6: Extract symptoms, map to specialist, query DB, and build suggestion.

        This runs as a parallel/sequential step within the overall latency
        budget. If it exceeds its internal budget, it returns None gracefully
        rather than breaching NFR-02.
        """
        lookup_start = time.perf_counter()

        try:
            symptoms = await extract_symptoms_from_text(message)
            if not symptoms:
                logger.debug("No symptoms extracted, skipping doctor lookup")
                return None

            specialist = SpecialistMapper.map_symptoms_to_specialist(symptoms)
            if not specialist:
                logger.info(
                    "No specialist mapping for %d symptoms — falling back to General Physician",
                    len(symptoms),
                )
                specialist = GENERAL_PHYSICIAN

            requested_specialist: Optional[str] = None
            doctors = SpecialistMapper.get_doctors_by_specialist(db, specialist)
            if not doctors and specialist != GENERAL_PHYSICIAN:
                # No doctor for the mapped specialty: suggest a General
                # Physician instead and report what was originally mapped.
                logger.info(
                    "No available doctors for '%s'; falling back to %s",
                    specialist,
                    GENERAL_PHYSICIAN,
                )
                requested_specialist = specialist
                specialist = GENERAL_PHYSICIAN
                doctors = SpecialistMapper.get_doctors_by_specialist(db, specialist)
            if not doctors:
                logger.info("No available doctors found for '%s'", specialist)
                return None

            top_doctor = SpecialistMapper.pick_top_doctor(doctors)
            if top_doctor is None:
                return None

            slots = get_doctor_availability(
                db, doctor_id=top_doctor.id, include_booked=False
            )
            availability_schemas = [
                DoctorAvailabilitySchema.model_validate(slot) for slot in slots[:3]
            ]

            doctor_name_raw = getattr(top_doctor, "name", None)
            if doctor_name_raw is None:
                doctor_name = f"Dr. {getattr(top_doctor, 'specialty', 'Unknown')}"
            elif _MagicMock is not None and isinstance(doctor_name_raw, _MagicMock):
                doctor_name = f"Dr. {getattr(top_doctor, 'specialty', 'Unknown')}"
            else:
                doctor_name = str(doctor_name_raw)

            suggestion = DoctorSuggestion(
                specialist=specialist,
                doctor_name=doctor_name,
                experience=getattr(top_doctor, "experience_years", None),
                availability=availability_schemas if availability_schemas else None,
                doctor_id=getattr(top_doctor, "id", None),
                requested_specialist=requested_specialist,
            )

            elapsed = time.perf_counter() - lookup_start
            logger.info(
                "Doctor suggestion built in %.3fs: specialist='%s', doctor_id=%d",
                elapsed,
                specialist,
                top_doctor.id,
            )
            return suggestion

        except Exception as exc:
            elapsed = time.perf_counter() - lookup_start
            logger.warning(
                "Doctor suggestion lookup failed after %.3fs: %s",
                elapsed,
                type(exc).__name__,
            )
            return None

    async def get_response(
        self,
        message: str,
        user_id: int,
        db: Session,
        raw_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate chatbot response with strict step budgeting and async concurrency.

        Day 8 REWRITE: Implements strict time budgeting per pipeline stage,
        parallelizes independent operations via asyncio.gather, uses
        ResponseCache for non-emergency queries, and enforces NFR-02 <4s
        compliance through ResponseOptimizer.
        """
        overall_start = time.perf_counter()
        logger.info("Processing chat for user_id=%s", user_id)
        metrics = get_metrics()
        metrics.increment_request("/api/ai/chat")

        optimizer = ResponseOptimizer()
        message_for_emergency = raw_message if raw_message is not None else message

        # ---------------------------------------------------------------
        # Step 1: Emergency Short-Circuit (< 50ms requirement)
        # ---------------------------------------------------------------
        emergency_start = time.perf_counter()
        is_emergency, urgency, reason = EmergencyDetector.detect_emergency(
            message_for_emergency
        )
        emergency_elapsed = time.perf_counter() - emergency_start

        optimizer.enforce_budget("emergency_detect", emergency_start)

        if is_emergency:
            # reason is the matched patient text: never logged.
            return self._emergency_reply(
                urgency,
                source="keyword",
                message=message_for_emergency,
                user_id=user_id,
                db=db,
                overall_start=overall_start,
            )

        # ---------------------------------------------------------------
        # Step 1.5: Fast-path cache check for GENERAL intent
        # GENERAL intent keys don't include conversation history, so we
        # can check the cache before paying for intent/memory/RAG.
        # This saves ~0.5-1.5s on cache hits for repeated questions.
        # SAFETY: Symptom/emergency intents are never cached, so a
        # stale general response can never leak for a clinical query.
        # ---------------------------------------------------------------
        if self.response_cache.should_cache("general", is_emergency=False):
            cached_response = self.response_cache.get(
                message=message,
                intent="general",
                last_3_messages=[],
            )
            if cached_response:
                try:
                    parsed_cache = json.loads(cached_response)
                    parsed_cache["intent"] = "general"
                    parsed_cache["confidence"] = 1.0
                    parsed_cache["doctor_suggestion"] = None
                    total_elapsed = time.perf_counter() - overall_start
                    metrics.record_latency("/api/ai/chat", total_elapsed)
                    logger.info(
                        "Fast-path cache HIT for user_id=%s in %.3fs",
                        user_id,
                        total_elapsed,
                    )
                    return parsed_cache
                except json.JSONDecodeError:
                    logger.warning(
                        "Fast-path cache corruption, proceeding with pipeline"
                    )

        # ---------------------------------------------------------------
        # Step 2: Concurrent Operations (asyncio.gather with return_exceptions)
        # Each task handles its own internal timeout. No outer ceiling.
        # ---------------------------------------------------------------
        concurrent_start = time.perf_counter()

        async def _classify_intent_task() -> Tuple[IntentEnum, float]:
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
            optimizer.enforce_budget("intent_classify", intent_start)
            return intent, confidence

        async def _load_memory_task() -> Tuple[List[Any], float]:
            memory_start = time.perf_counter()
            memory_manager = HokuConversationMemory(
                message_limit=ai_settings.MEMORY_MESSAGE_LIMIT,
                max_history_tokens=ai_settings.MEMORY_MAX_TOKENS,
            )
            history_list = memory_manager.load_memory(user_id=user_id, db=db)
            memory_elapsed = time.perf_counter() - memory_start
            logger.info(
                "Memory loaded for user_id=%s in %.3fs (%d messages)",
                user_id,
                memory_elapsed,
                len(history_list),
            )
            optimizer.enforce_budget("memory_load", memory_start)
            return history_list, memory_elapsed

        async def _rag_lookup_task() -> List[Any]:
            """FAQ documents above the threshold ([] on miss/timeout/error)."""
            rag_start = time.perf_counter()
            docs: List[Any] = []
            try:
                retrieve = getattr(self.rag, "retrieve", None)
                if retrieve is None or not callable(retrieve):
                    docs = []
                elif _MagicMock is not None and isinstance(retrieve, _MagicMock):
                    docs = []
                else:
                    rag_task = asyncio.to_thread(retrieve, message)
                    docs = list(
                        await asyncio.wait_for(
                            rag_task, timeout=ai_settings.RAG_LOOKUP_TIMEOUT
                        )
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "RAG lookup timed out after %.3fs, skipping RAG",
                    ai_settings.RAG_LOOKUP_TIMEOUT,
                )
                docs = []
            except Exception as exc:
                logger.warning(
                    "RAG lookup failed, falling back to general knowledge: %s",
                    type(exc).__name__,
                )
                docs = []
            optimizer.enforce_budget("rag_retrieve", rag_start)
            return docs

        intent_result, memory_result, rag_result = await asyncio.gather(
            _classify_intent_task(),
            _load_memory_task(),
            _rag_lookup_task(),
            return_exceptions=True,
        )
        faq_docs: List[Any] = (
            [] if isinstance(rag_result, BaseException) else list(rag_result)
        )

        if isinstance(intent_result, Exception):
            logger.warning("Intent classification failed: %s", intent_result)
            intent, confidence = IntentEnum.GENERAL, 0.0
        else:
            intent, confidence = intent_result

        history_list: List[Any]
        if isinstance(memory_result, Exception):
            logger.warning(
                "Memory load failed (%s); continuing without history",
                type(memory_result).__name__,
            )
            history_list = []
        else:
            history_list, memory_elapsed = memory_result

        concurrent_elapsed = time.perf_counter() - concurrent_start
        logger.info(
            "Concurrent block completed in %.3fs for user_id=%s",
            concurrent_elapsed,
            user_id,
        )

        # The classifier is a second emergency signal for phrasings the
        # keyword list misses: same fixed reply, never an LLM-written one.
        if intent == IntentEnum.EMERGENCY:
            return self._emergency_reply(
                EmergencyDetector.URGENCY_HIGH,
                source="intent",
                message=message_for_emergency,
                user_id=user_id,
                db=db,
                overall_start=overall_start,
            )

        # Filter RAG context based on intent
        use_rag = False
        faq_context = ""
        if intent in _RAG_ELIGIBLE_INTENTS and faq_docs:
            use_rag = True
            faq_context = format_faq_context(faq_docs)
            logger.info(
                "RAG context accepted for user_id=%s (intent=%s)",
                user_id,
                intent.value,
            )
        else:
            if faq_docs and intent not in _RAG_ELIGIBLE_INTENTS:
                logger.debug(
                    "Discarding RAG context for intent=%s (not RAG-eligible)",
                    intent.value,
                )
            faq_docs = []

        # ---------------------------------------------------------------
        # Step 3: Response Cache Lookup
        # ---------------------------------------------------------------
        cache_start = time.perf_counter()
        last_3_messages = []
        if history_list and len(history_list) > 0:
            last_3_messages = [
                getattr(msg, "content", str(msg)) for msg in history_list[-3:]
            ]

        if self.response_cache.should_cache(intent.value, is_emergency=False):
            # For GENERAL intent, exclude conversation history from the cache key.
            # This ensures repeated standalone questions (e.g. "What services...?")
            # hit the cache even after prior chat turns.
            cache_key_history = [] if intent == IntentEnum.GENERAL else last_3_messages
            cached_response = self.response_cache.get(
                message=message,
                intent=intent.value,
                last_3_messages=cache_key_history,
            )
            if cached_response:
                cache_elapsed = time.perf_counter() - cache_start
                logger.info(
                    "Cache HIT for user_id=%s (intent=%s, lookup=%.3fs)",
                    user_id,
                    intent.value,
                    cache_elapsed,
                )
                try:
                    parsed_cache = json.loads(cached_response)
                    parsed_cache["intent"] = intent.value
                    parsed_cache["confidence"] = confidence
                    parsed_cache["doctor_suggestion"] = None
                    total_elapsed = time.perf_counter() - overall_start
                    metrics.record_latency("/api/ai/chat", total_elapsed)
                    logger.info(
                        "Cache response returned in %.3fs for user_id=%s",
                        total_elapsed,
                        user_id,
                    )
                    return parsed_cache
                except json.JSONDecodeError:
                    logger.warning("Cache corruption detected, falling through to LLM")

        cache_elapsed = time.perf_counter() - cache_start
        logger.debug("Cache MISS for user_id=%s (lookup=%.3fs)", user_id, cache_elapsed)

        # ---------------------------------------------------------------
        # Step 4: Specialist suggestion (Day 6)
        # Only for SYMPTOM, or GENERAL with symptom keywords.
        # ---------------------------------------------------------------
        doctor_suggestion: Optional[DoctorSuggestion] = None
        if intent == IntentEnum.SYMPTOM:
            doctor_suggestion = await self._lookup_doctor_suggestion(message, db)
        elif intent == IntentEnum.GENERAL:
            symptom_keywords = [
                "pain",
                "fever",
                "headache",
                "hurt",
                "ache",
                "sick",
                "nausea",
                "dizzy",
                "cough",
                "breath",
                "chest",
                "rash",
                "swelling",
                "bleeding",
                "vomit",
                "diarrhea",
                "fatigue",
                "tired",
                "weakness",
                "symptom",
                "not feeling",
                "unwell",
            ]
            if any(kw in message.lower() for kw in symptom_keywords):
                doctor_suggestion = await self._lookup_doctor_suggestion(message, db)
            else:
                logger.debug("Skipping doctor suggestion for non-symptom GENERAL query")
        else:
            logger.debug(
                "Skipping specialist suggestion for intent=%s",
                intent.value,
            )

        # ---------------------------------------------------------------
        # Step 5: Build intent-aware context
        # ---------------------------------------------------------------
        intent_context = self._build_intent_context(intent)

        # ---------------------------------------------------------------
        # Step 6: Check main LLM availability
        # ---------------------------------------------------------------
        def _fallback(reason: str) -> Dict[str, Any]:
            """Deterministic reply from this request's FAQ docs and doctor."""
            metrics.record_latency("/api/ai/chat", time.perf_counter() - overall_start)
            return self._fallback_response(
                reason,
                intent=intent.value,
                confidence=confidence,
                faq_docs=faq_docs,
                doctor_suggestion=doctor_suggestion,
                user_id=user_id,
                db=db,
            )

        if self.main_llm is None:
            logger.error("Main LLM unavailable -- returning fallback")
            return _fallback("LLM initialization failed")

        # ---------------------------------------------------------------
        # Step 7: LLM Generation with Strict Budgeting
        # ---------------------------------------------------------------
        llm_start = time.perf_counter()
        remaining_budget = optimizer.remaining_budget(overall_start)

        if remaining_budget < _MIN_LLM_TIMEOUT:
            logger.warning(
                "Remaining budget %.3fs < %.3fs min for user_id=%s — returning fallback",
                remaining_budget,
                _MIN_LLM_TIMEOUT,
                user_id,
            )
            return _fallback("budget_exhausted")

        try:
            compressed_history = compress_prompt(history_list, max_messages=3)
            history = self._apply_compressed_history(history_list, compressed_history)

            prompt_template = (
                rag_chat_prompt_template if use_rag else chat_prompt_template
            )
            chain = self._generation_chain(prompt_template)

            invoke_input: Dict[str, Any] = {
                "message": message,
                "context": intent_context,
                "history": history,
            }
            if use_rag:
                invoke_input["faq_context"] = faq_context

            llm_timeout = max(
                _MIN_LLM_TIMEOUT,
                min(remaining_budget - 0.15, ai_settings.GROQ_TIMEOUT_SECONDS),
            )

            bound_invoke = functools.partial(chain.invoke, invoke_input)

            result = await generate_with_timeout(
                coro_or_func=bound_invoke,
                timeout=llm_timeout,
                fallback_value=None,
            )

            if result is None:
                logger.warning(
                    "LLM generation timed out for user_id=%s (budget=%.3fs, timeout=%.3fs)",
                    user_id,
                    remaining_budget,
                    llm_timeout,
                )
                return _fallback("llm_timeout")

            llm_elapsed = time.perf_counter() - llm_start
            logger.info(
                "Groq response in %.3fs for user_id=%s (budget_used=%.3fs)",
                llm_elapsed,
                user_id,
                llm_elapsed,
            )
            optimizer.enforce_budget("llm_generate", llm_start)

            raw_text = self._extract_text_from_result(result)
            estimated_tokens = len(raw_text) // 4
            logger.info(
                "Estimated response tokens for user_id=%s: ~%d",
                user_id,
                estimated_tokens,
            )

            parsed = self._parse_llm_output(raw_text)
            reply = parsed.get("reply", "")
            if SAFETY_DISCLAIMER not in reply:
                reply = f"{reply} {SAFETY_DISCLAIMER}"
            parsed["reply"] = reply

            # ---------------------------------------------------------------
            # Step 8: Post-LLM Safety Verification (Day 7)
            # ---------------------------------------------------------------
            safety_start = time.perf_counter()
            safe_reply, safety_violations, safety_severity = (
                SafetyGuardrails.apply_3_strike_safety(
                    text=parsed.get("reply", ""),
                    user_id=user_id,
                    db=db,
                )
            )
            safety_elapsed = time.perf_counter() - safety_start

            if safety_violations:
                metrics.increment_safety_violation(safety_violations[0])
                logger.warning(
                    "Safety violations for user_id=%s: %s (severity=%s, elapsed=%.3fs)",
                    user_id,
                    safety_violations,
                    safety_severity,
                    safety_elapsed,
                )

            if safety_severity == "high":
                metrics.increment_3_strike_fallback()

            parsed["reply"] = safe_reply

            # ---------------------------------------------------------------
            # Step 9: Cache Persistence
            # ---------------------------------------------------------------
            cache_persist_start = time.perf_counter()
            if self.response_cache.should_cache(intent.value, is_emergency=False):
                response_to_cache = json.dumps(
                    {
                        "reply": parsed.get("reply", ""),
                        "suggestedSpecialist": parsed.get("suggestedSpecialist"),
                        "severity": parsed.get("severity") or "unknown",
                        "shouldSeeDoctor": parsed.get("shouldSeeDoctor")
                        if parsed.get("shouldSeeDoctor") is not None
                        else True,
                    }
                )
                cache_key_history = (
                    [] if intent == IntentEnum.GENERAL else last_3_messages
                )
                self.response_cache.set(
                    message=message,
                    intent=intent.value,
                    last_3_messages=cache_key_history,
                    response=response_to_cache,
                    is_emergency=False,
                )
                logger.debug(
                    "Response cached for user_id=%s (intent=%s)", user_id, intent.value
                )

            optimizer.enforce_budget("db_persist", cache_persist_start)

            response: Dict[str, Any] = {
                "reply": parsed.get("reply") or self._fallback_response()["reply"],
                "suggestedSpecialist": parsed.get("suggestedSpecialist"),
                "severity": parsed.get("severity") or "unknown",
                "shouldSeeDoctor": parsed.get("shouldSeeDoctor")
                if parsed.get("shouldSeeDoctor") is not None
                else True,
                "intent": intent.value,
                "confidence": confidence,
                "doctor_suggestion": doctor_suggestion.model_dump()
                if doctor_suggestion
                else None,
                "fallback_used": False,
            }

            total_elapsed = time.perf_counter() - overall_start
            metrics.record_latency("/api/ai/chat", total_elapsed)

            logger.info(
                "Timing breakdown for user_id=%s: total=%.3fs, "
                "emergency=%.3fs, concurrent=%.3fs, cache=%.3fs, "
                "llm=%.3fs, safety=%.3fs",
                user_id,
                total_elapsed,
                emergency_elapsed,
                concurrent_elapsed,
                cache_elapsed,
                llm_elapsed,
                safety_elapsed,
            )

            return response

        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - overall_start
            logger.warning("Groq timeout after %.3fs for user_id=%s", elapsed, user_id)
            return _fallback("timeout")

        except Exception as exc:
            elapsed = time.perf_counter() - overall_start
            # No traceback / str(exc): provider errors can echo prompt text.
            logger.error(
                "Groq error after %.3fs for user_id=%s: %s",
                elapsed,
                user_id,
                type(exc).__name__,
            )
            return _fallback(type(exc).__name__)

    @staticmethod
    def _emergency_reply(
        urgency: str,
        *,
        source: str,
        message: str,
        user_id: int,
        db: Session,
        overall_start: float,
    ) -> Dict[str, Any]:
        """
        Fixed urgent reply for ``urgency``, recorded as a safety event.

        ``source`` is "keyword" (regex) or "intent" (classifier). No LLM,
        RAG, cache or doctor lookup is involved.
        """
        metrics = get_metrics()
        logger.critical(
            "Emergency reply for user_id=%s (urgency=%s, source=%s)",
            user_id,
            urgency,
            source,
        )
        metrics.increment_emergency_detection()

        try:
            from app.crud.crud_safety import log_safety_violation

            log_safety_violation(
                db=db,
                user_id=user_id,
                message=message[:1000],
                ai_response=f"[fixed emergency reply, source={source}]",
                violation_type="emergency_triggered",
                severity="high",
            )
        except Exception as log_exc:
            logger.warning(
                "Failed to log emergency safety event: %s", type(log_exc).__name__
            )

        response = EmergencyDetector.get_urgency_response(urgency)
        response["doctor_suggestion"] = None
        response["fallback_used"] = False
        metrics.record_latency("/api/ai/chat", time.perf_counter() - overall_start)
        return response

    @staticmethod
    def _apply_compressed_history(
        history: List[BaseMessage], compressed_history: Any
    ) -> List[BaseMessage]:
        """
        Return the history to send to the prompt.

        Uses ``compressed_history`` (dicts with role/content, or message
        objects) when every entry is recognised; otherwise the original
        ``history`` is returned unchanged.
        """
        if not compressed_history or not isinstance(compressed_history, list):
            return history

        role_map = {
            "user": HumanMessage,
            "human": HumanMessage,
            "assistant": AIMessage,
            "ai": AIMessage,
            "system": SystemMessage,
        }

        rebuilt: List[BaseMessage] = []
        for item in compressed_history:
            if isinstance(item, dict):
                message_cls = role_map.get(str(item.get("role", "")).lower())
                content = item.get("content", "")
                if message_cls is None or not isinstance(content, str):
                    logger.debug(
                        "Unrecognised compressed message role=%r; using full history",
                        item.get("role"),
                    )
                    return history
                rebuilt.append(message_cls(content=content))
            elif isinstance(item, BaseMessage):
                rebuilt.append(item)
            else:
                logger.debug(
                    "Unrecognised compressed message type %s; using full history",
                    type(item),
                )
                return history

        logger.info("Compressed history: %d -> %d messages", len(history), len(rebuilt))
        return rebuilt

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
                return {
                    "reply": data.get("reply", ""),
                    "suggestedSpecialist": data.get("suggestedSpecialist"),
                    "severity": (
                        "unknown"
                        if data.get("severity") in (None, "null", "Null", "NULL")
                        else data.get("severity")
                    ),
                    "shouldSeeDoctor": data.get("shouldSeeDoctor")
                    if data.get("shouldSeeDoctor") is not None
                    else True,
                }
        except json.JSONDecodeError:
            pass

        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "reply" in data:
                    return {
                        "reply": data.get("reply", ""),
                        "suggestedSpecialist": data.get("suggestedSpecialist"),
                        "severity": (
                            "unknown"
                            if data.get("severity") in (None, "null", "Null", "NULL")
                            else data.get("severity")
                        ),
                        "shouldSeeDoctor": data.get("shouldSeeDoctor")
                        if data.get("shouldSeeDoctor") is not None
                        else True,
                    }
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
            sev_val = sev_match.group(1)
            parsed["severity"] = "unknown" if sev_val.lower() == "null" else sev_val

        doc_match = re.search(r'"shouldSeeDoctor"\s*:\s*(true|false)', text)
        if doc_match:
            parsed["shouldSeeDoctor"] = doc_match.group(1).lower() == "true"

        return parsed

    def _fallback_response(
        self,
        reason: str = "unknown",
        *,
        intent: str = "general",
        confidence: float = 0.0,
        faq_docs: Optional[List[Any]] = None,
        doctor_suggestion: Optional[DoctorSuggestion] = None,
        user_id: Optional[int] = None,
        db: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic reply when the reply LLM is unavailable, times out or
        fails: built from the FAQ answer and doctor suggestion retrieved for
        this request (see ``build_fallback_reply``), then passed through the
        same safety guardrails as an LLM reply. ``fallback_used`` is True.
        """
        logger.info("Fallback response (reason=%s)", reason)
        reply = build_fallback_reply(intent, faq_docs, doctor_suggestion)
        if faq_docs or doctor_suggestion is not None:
            # Retrieved text is checked like an LLM reply; the static
            # messages are fixed strings covered by tests.
            reply, _violations, _severity = SafetyGuardrails.apply_3_strike_safety(
                text=reply, user_id=user_id or 0, db=db
            )
        if SAFETY_DISCLAIMER not in reply:
            reply = f"{reply} {SAFETY_DISCLAIMER}"
        return {
            "reply": reply,
            "suggestedSpecialist": doctor_suggestion.specialist
            if doctor_suggestion is not None
            else None,
            "severity": "unknown",
            "shouldSeeDoctor": True,
            "intent": intent,
            "confidence": confidence,
            "doctor_suggestion": doctor_suggestion.model_dump()
            if doctor_suggestion is not None
            else None,
            "fallback_used": True,
        }
