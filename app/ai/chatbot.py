"""
Hoku Health Care - AI Chatbot Engine (Day 3: Conversation Memory).

Core chatbot logic using Groq LLMs via LangChain 0.2.6 with per-user
conversation memory loaded from PostgreSQL/SQLite.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any, Dict, Optional

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
    logging.getLogger(__name__).warning(
        "LangChain/Groq not installed: %s", _import_exc
    )

from app.ai.config import ai_settings
from app.ai.memory import HokuConversationMemory
from app.ai.prompts import chat_prompt_template
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)


class HokuChatbot:
    """
    Hoku Health Care AI Chatbot.

    Uses Groq via LangChain LLMChain:
    - Fast model (llama-3.1-8b-instant): Intent classification (Day 4).
    - Main model (llama-3.3-70b-versatile): Patient-facing response.
    """

    def __init__(self) -> None:
        """Initialize with lazy-loaded Groq clients."""
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

    def build_chain(self) -> Any:
        """Build LLMChain with system prompt and main model."""
        if self._chain is None:
            if LLMChain is None:
                raise RuntimeError("LLMChain not available")
            self._chain = LLMChain(
                llm=self.main_llm,
                prompt=chat_prompt_template,
                verbose=False,
            )
            logger.info("LLMChain built with model=%s", self.main_model)
        return self._chain

    async def get_response(
        self,
        message: str,
        user_id: int,
        db: Session,
    ) -> Dict[str, Any]:
        """
        Generate chatbot response with timeout, fallback, and memory.

        Args:
            message: Sanitized user message.
            user_id: Authenticated user ID.
            db: SQLAlchemy database session for memory loading.

        Returns:
            Dict with reply, suggestedSpecialist, severity, shouldSeeDoctor.
        """
        start_time = time.perf_counter()
        logger.info("Processing chat for user_id=%s", user_id)

        if self.main_llm is None:
            logger.error("Main LLM unavailable — returning fallback")
            return self._fallback_response("LLM initialization failed")

        try:
            # ------------------------------------------------------------------
            # Day 3: Load conversation memory from database
            # ------------------------------------------------------------------
            memory_start = time.perf_counter()
            memory_manager = HokuConversationMemory(
                message_limit=ai_settings.MEMORY_MESSAGE_LIMIT,
                max_history_tokens=ai_settings.MEMORY_MAX_TOKENS,
            )
            memory = memory_manager.load_memory(user_id=user_id, db=db)
            memory_elapsed = time.perf_counter() - memory_start

            # Log memory state for debugging
            memory_vars = memory.load_memory_variables({"message": message})
            history_list = memory_vars.get("history", [])
            logger.info(
                "Memory loaded for user_id=%s in %.3fs (%d messages)",
                user_id,
                memory_elapsed,
                len(history_list),
            )

            # Adjust LLM timeout so memory load + LLM stays under 3.5s total
            remaining_timeout = max(0.1, self.timeout - memory_elapsed)

            # Build chain WITH memory (Day 3 requirement: LLMChain(..., memory=memory))
            # MessagesPlaceholder + ConversationBufferMemory(return_messages=True)
            # handles history injection automatically. Do NOT pass 'history'
            # explicitly in chain.invoke() — it causes key collision.
            if LLMChain is None:
                raise RuntimeError("LLMChain not available")

            chain = LLMChain(
                llm=self.main_llm,
                prompt=chat_prompt_template,
                memory=memory,
                verbose=False,
            )

            # TODO Day 4: Use fast_llm for intent classification
            # intent = await self._classify_intent(message)

            llm_task = asyncio.to_thread(
                chain.invoke,
                {"message": message, "context": "No additional context available."}
            )
            result = await asyncio.wait_for(llm_task, timeout=remaining_timeout)

            elapsed = time.perf_counter() - start_time
            logger.info("Groq response in %.3fs for user_id=%s", elapsed, user_id)

            # Log estimated token usage for cost monitoring
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

            return {
                "reply": parsed.get("reply", self._fallback_response()["reply"]),
                "suggestedSpecialist": parsed.get("suggestedSpecialist"),
                "severity": parsed.get("severity", "unknown"),
                "shouldSeeDoctor": parsed.get("shouldSeeDoctor", True),
            }

        except asyncio.TimeoutError:
            elapsed = time.perf_counter() - start_time
            logger.warning("Groq timeout after %.3fs for user_id=%s", elapsed, user_id)
            return self._fallback_response("timeout")

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.exception(
                "Groq error after %.3fs for user_id=%s: %s", elapsed, user_id, exc
            )
            return self._fallback_response(str(exc))

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
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
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
        }