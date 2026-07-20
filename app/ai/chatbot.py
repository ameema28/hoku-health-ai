"""
Hoku Health Care - AI Chatbot Engine (Day 6: Specialist suggestion & doctor integration).

Core chatbot logic using Groq LLMs via LangChain 0.2.6 with per-user
conversation memory, intent classification, emergency detection, RAG,
and (Day 6) symptom-to-specialist mapping with doctor suggestions.

Flow:
1. Emergency detection (regex, <50ms) -- bypasses LLM/RAG if emergency
2. Intent classification (llama-3.1-8b-instant, <500ms)
3. RAG lookup (Day 5): for GENERAL/SYMPTOM intents only
4. Symptom extraction + specialist mapping + doctor lookup (Day 6)
5. Intent-aware system prompt augmentation
6. Main LLM response generation (llama-3.3-70b-versatile)
7. Persistence with intent metadata
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

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
from app.ai.specialist_mapper import SpecialistMapper
from app.ai.symptom_extractor import extract_symptoms_from_text
from app.crud.crud_doctor import get_doctor_availability
from app.schemas.schemas_doctor import DoctorAvailability as DoctorAvailabilitySchema, DoctorSuggestion
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# Day 5: intents for which a Hoku FAQ lookup is worthwhile.
_RAG_ELIGIBLE_INTENTS = {IntentEnum.GENERAL, IntentEnum.SYMPTOM}

# Day 6: intents eligible for specialist suggestion flow
_SPECIALIST_ELIGIBLE_INTENTS = {IntentEnum.SYMPTOM, IntentEnum.GENERAL}


class HokuChatbot:
    """
    Hoku Health Care AI Chatbot.

    Uses Groq via LangChain LLMChain:
    - Fast model (llama-3.1-8b-instant): Intent classification (Day 4).
    - Main model (llama-3.3-70b-versatile): Patient-facing response.

    Day 5 addition: Lazily-initialized HokuRAG instance.
    Day 6 addition: SpecialistMapper for symptom-to-doctor routing.
    """

    def __init__(self) -> None:
        """Initialize with lazy-loaded Groq clients, intent classifier, RAG, and specialist mapper."""
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
        self._rag: Optional[HokuRAG] = None  # Day 5
        self._specialist_mapper: Optional[SpecialistMapper] = None  # Day 6

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

    @property
    def specialist_mapper(self) -> SpecialistMapper:
        """Lazy initializer for the Day 6 specialist mapper."""
        if self._specialist_mapper is None:
            self._specialist_mapper = SpecialistMapper()
        return self._specialist_mapper

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

        Args:
            message: Sanitized user message.
            db: SQLAlchemy database session.

        Returns:
            DoctorSuggestion | None: Suggestion data or None if lookup fails.
        """
        lookup_start = time.perf_counter()

        try:
            # Extract symptoms (fast regex or LLM fallback with 0.2s timeout)
            symptoms = await extract_symptoms_from_text(message)
            if not symptoms:
                logger.debug("No symptoms extracted, skipping doctor lookup")
                return None

            # Map to specialist
            specialist = SpecialistMapper.map_symptoms_to_specialist(symptoms)
            if not specialist:
                logger.debug("No specialist mapping found, skipping doctor lookup")
                return None

            # Query DB for available doctors
            doctors = SpecialistMapper.get_doctors_by_specialist(db, specialist)
            if not doctors:
                logger.info("No available doctors found for '%s'", specialist)
                return None

            # Pick top doctor
            top_doctor = SpecialistMapper.pick_top_doctor(doctors)
            if top_doctor is None:
                return None

            # Get availability (limit to next 3 slots for brevity)
            slots = get_doctor_availability(db, doctor_id=top_doctor.id, include_booked=False)
            availability_schemas = [
                DoctorAvailabilitySchema.model_validate(slot) for slot in slots[:3]
            ]

            suggestion = DoctorSuggestion(
                specialist=specialist,
                doctor_name=getattr(top_doctor, "name", f"Dr. {top_doctor.specialty}"),
                experience=top_doctor.experience_years,
                availability=availability_schemas if availability_schemas else None,
                doctor_id=top_doctor.id,
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
                exc,
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
        Generate chatbot response with intent classification, emergency
        detection, RAG retrieval, specialist suggestion, timeout, fallback,
        and memory.

        Steps:
        1. Emergency detection on RAW message (regex, <50ms)
        2. Intent classification on sanitized message (async, <500ms)
        3. RAG lookup for GENERAL/SYMPTOM intents (Day 5)
        4. Specialist suggestion for SYMPTOM/GENERAL intents (Day 6)
        5. Build intent-aware context
        6. Load conversation memory from DB
        7. Generate response with main LLM
        8. Return with intent, confidence, and doctor_suggestion metadata
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
                "No RAG context for user_id=%s (intent=%s, %.3fs)",
                user_id,
                intent.value,
                rag_elapsed,
            )

        # ---------------------------------------------------------------
        # Step 4: Specialist suggestion (Day 6) -- SYMPTOM/GENERAL only
        # Skip for BOOKING, MEDICATION, EMERGENCY
        # ---------------------------------------------------------------
        doctor_suggestion: Optional[DoctorSuggestion] = None
        if intent in _SPECIALIST_ELIGIBLE_INTENTS:
            doctor_suggestion = await self._lookup_doctor_suggestion(message, db)
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
        if self.main_llm is None:
            logger.error("Main LLM unavailable -- returning fallback")
            fallback = self._fallback_response("LLM initialization failed")
            fallback["intent"] = intent.value
            fallback["confidence"] = confidence
            fallback["doctor_suggestion"] = None
            return fallback

        try:
            # -------------------------------------------------------
            # Step 7: Load conversation memory from database
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

            # Adjust LLM timeout
            remaining_timeout = max(
                0.1,
                self.timeout - intent_elapsed - rag_elapsed - memory_elapsed,
            )

            # -------------------------------------------------------
            # Step 8: Build chain and generate response
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

            response: Dict[str, Any] = {
                "reply": parsed.get("reply", self._fallback_response()["reply"]),
                "suggestedSpecialist": parsed.get("suggestedSpecialist"),
                "severity": parsed.get("severity", "unknown"),
                "shouldSeeDoctor": parsed.get("shouldSeeDoctor", True),
                "intent": intent.value,
                "confidence": confidence,
                "doctor_suggestion": doctor_suggestion.model_dump() if doctor_suggestion else None,
            }
            return response

        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.warning("Groq timeout after %.3fs for user_id=%s", elapsed, user_id)
            fallback = self._fallback_response("timeout")
            fallback["intent"] = intent.value
            fallback["confidence"] = confidence
            fallback["doctor_suggestion"] = None
            return fallback

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception("Groq error after %.3fs for user_id=%s: %s", elapsed, user_id, exc)
            fallback = self._fallback_response(str(exc))
            fallback["intent"] = intent.value
            fallback["confidence"] = confidence
            fallback["doctor_suggestion"] = None
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
            "doctor_suggestion": None,
        }