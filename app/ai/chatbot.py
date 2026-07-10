"""
Hoku Health Care - AI Chatbot Engine.

Core chatbot logic using Groq LLMs via LangChain. Implements a two-model
strategy for cost-efficient intent recognition and high-quality response
generation. All outputs include a mandatory clinical safety disclaimer.
"""

import logging
from typing import Any, Dict, Optional

from app.core.config import settings
from app.utils.constants import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)


class HokuChatbot:
    """
    Hoku Health Care AI Chatbot.

    Uses a dual-model strategy with Groq:
    - Fast model (llama3-8b-8192): Intent classification and entity extraction.
    - Main model (llama3-70b-8192 or mixtral): Final patient-facing response.

    This balances latency (NFR-02: <4s) with response quality.
    """

    def __init__(self) -> None:
        """
        Initialize the chatbot with Groq LLM clients.

        Loads API key from settings and configures both fast and main
        language models. Falls back to mock responses if Groq is unavailable.
        """
        self.groq_api_key = settings.GROQ_API_KEY
        self.fast_model = settings.GROQ_FAST_MODEL
        self.main_model = settings.GROQ_MAIN_MODEL

        # Lazy-load LangChain clients on first use to reduce startup time
        self._fast_llm: Optional[Any] = None
        self._main_llm: Optional[Any] = None

    @property
    def fast_llm(self) -> Any:
        """Lazy initializer for the fast classification model."""
        if self._fast_llm is None:
            try:
                from langchain_groq import ChatGroq
                self._fast_llm = ChatGroq(
                    model=self.fast_model,
                    api_key=self.groq_api_key,
                    temperature=0.0,
                    max_tokens=256,
                )
                logger.info("Fast LLM (%s) initialized", self.fast_model)
            except Exception as exc:
                logger.warning("Failed to initialize fast LLM: %s", exc)
                self._fast_llm = None
        return self._fast_llm

    @property
    def main_llm(self) -> Any:
        """Lazy initializer for the main response generation model."""
        if self._main_llm is None:
            try:
                from langchain_groq import ChatGroq
                self._main_llm = ChatGroq(
                    model=self.main_model,
                    api_key=self.groq_api_key,
                    temperature=0.7,
                    max_tokens=1024,
                )
                logger.info("Main LLM (%s) initialized", self.main_model)
            except Exception as exc:
                logger.warning("Failed to initialize main LLM: %s", exc)
                self._main_llm = None
        return self._main_llm

    async def get_response(self, message: str, user_id: int) -> Dict[str, Any]:
        """
        Generate a chatbot response for the given user message.

        Workflow:
        1. (Future) Classify intent using fast model.
        2. (Future) Retrieve relevant health FAQs via RAG (pgvector).
        3. Generate safe, empathetic response using main model.
        4. Append mandatory safety disclaimer.

        Args:
            message: Sanitized user message.
            user_id: Authenticated user ID for context.

        Returns:
            Dict[str, Any]: Response dict matching ChatMessageResponse schema.
        """
        # Stub implementation for setup day
        # TODO: Replace with actual LLM invocation after RAG pipeline is ready

        logger.info("Processing chat for user_id=%s", user_id)

        # Mock response for infrastructure validation
        reply = (
            "I understand your concern. I'm here to help with general health "
            "information and guidance. Could you tell me more about your symptoms "
            "so I can suggest the right specialist? "
            f"{SAFETY_DISCLAIMER}"
        )

        return {
            "reply": reply,
            "suggestedSpecialist": "General Physician",
            "severity": "mild",
            "shouldSeeDoctor": False,
        }
