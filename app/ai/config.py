"""
Hoku Health Care - AI Configuration Module (Day 7).

Centralizes all AI/LLM hyperparameters, timeouts, and retry policies.
All values are tuned for clinical safety and the <4s response NFR.

Day 4 additions:
- Intent classification settings (timeout, threshold, model)

Day 6 additions:
- Symptom extraction settings (timeout, model)
- Doctor lookup limit

Day 7 additions:
- Emergency check timeout (Tier 2 LLM fallback)
- Safety max retries (3-strike mechanism)
- Safety fallback response (hardcoded safe message)
"""

import logging
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.config import settings as core_settings

logger = logging.getLogger(__name__)


class AISettings(BaseSettings):
    """
    AI-specific settings layered on top of core application config.
    These parameters are deliberately conservative to balance response
    quality with the strict <4s latency requirement (NFR-02).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Model Selection
    # ------------------------------------------------------------------
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAIN_MODEL: str = "llama-3.3-70b-versatile"

    # ------------------------------------------------------------------
    # Generation Hyperparameters (Clinical Rationale)
    # ------------------------------------------------------------------
    TEMPERATURE: float = 0.3
    MAX_TOKENS: int = 512

    # ------------------------------------------------------------------
    # Timeout & Retry Policy
    # ------------------------------------------------------------------
    GROQ_TIMEOUT_SECONDS: float = 3.5
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE_SECONDS: float = 1.0

    # ------------------------------------------------------------------
    # Structured Output
    # ------------------------------------------------------------------
    RESPONSE_FORMAT: str = "json"

    # ------------------------------------------------------------------
    # Conversation Memory (Day 3)
    # ------------------------------------------------------------------
    MEMORY_MESSAGE_LIMIT: int = 10
    MEMORY_MAX_TOKENS: int = 307
    TIKTOKEN_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Intent Classification (Day 4)
    # ------------------------------------------------------------------
    INTENT_MODEL: str = "llama-3.1-8b-instant"
    INTENT_CLASSIFICATION_TIMEOUT: float = 0.5
    INTENT_CONFIDENCE_THRESHOLD: float = 0.7

    # ------------------------------------------------------------------
    # RAG Lookup (Day 5)
    # ------------------------------------------------------------------
    RAG_LOOKUP_TIMEOUT: float = 0.5

    # ------------------------------------------------------------------
    # Symptom Extraction & Doctor Lookup (Day 6)
    # ------------------------------------------------------------------
    # Timeout for LLM-based symptom extraction fallback (0.2s hard limit).
    # If exceeded, defaults to ["fever"] -> General Physician to protect NFR-02.
    SYMPTOM_EXTRACTION_TIMEOUT: float = 0.2

    # Model for LLM-based symptom extraction fallback.
    SYMPTOM_EXTRACTION_MODEL: str = "llama-3.1-8b-instant"

    # Maximum number of doctors to return in a specialist lookup.
    DOCTOR_LOOKUP_LIMIT: int = 5

    # ------------------------------------------------------------------
    # Emergency Detection & Safety Guardrails (Day 7)
    # ------------------------------------------------------------------
    # Timeout for Tier 2 LLM emergency check (ambiguous edge cases).
    # Tier 1 regex runs in <50ms regardless; this is only for fallback.
    EMERGENCY_CHECK_TIMEOUT: float = 0.3

    # Maximum safety retry attempts before returning hardcoded fallback.
    # Each retry sanitizes the response and re-validates.
    SAFETY_MAX_RETRIES: int = 3

    # Hardcoded safe fallback response when all safety retries fail.
    # This is the absolute last resort — never returns unsafe content.
    SAFETY_FALLBACK_RESPONSE: str = (
        "I am unable to provide a medical opinion for this query. "
        "Please consult a qualified doctor immediately."
    )

    @property
    def groq_api_key(self) -> str:
        """Delegate to core settings to keep secrets in one place."""
        return core_settings.GROQ_API_KEY

    @property
    def is_production(self) -> bool:
        """Delegate to core settings."""
        return core_settings.is_production


def get_ai_settings() -> AISettings:
    """Factory for AI settings with validation logging."""
    ai_settings = AISettings()
    logger.info(
        "AI settings loaded: model=%s, temp=%.1f, max_tokens=%d, timeout=%.1fs, "
        "memory_limit=%d, memory_tokens=%d, intent_model=%s, "
        "intent_timeout=%.2fs, intent_threshold=%.2f, "
        "symptom_timeout=%.2fs, doctor_limit=%d, "
        "emergency_timeout=%.2fs, safety_retries=%d",
        ai_settings.GROQ_MAIN_MODEL,
        ai_settings.TEMPERATURE,
        ai_settings.MAX_TOKENS,
        ai_settings.GROQ_TIMEOUT_SECONDS,
        ai_settings.MEMORY_MESSAGE_LIMIT,
        ai_settings.MEMORY_MAX_TOKENS,
        ai_settings.INTENT_MODEL,
        ai_settings.INTENT_CLASSIFICATION_TIMEOUT,
        ai_settings.INTENT_CONFIDENCE_THRESHOLD,
        ai_settings.SYMPTOM_EXTRACTION_TIMEOUT,
        ai_settings.DOCTOR_LOOKUP_LIMIT,
        ai_settings.EMERGENCY_CHECK_TIMEOUT,
        ai_settings.SAFETY_MAX_RETRIES,
    )
    return ai_settings


ai_settings = get_ai_settings()