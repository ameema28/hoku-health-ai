"""
Hoku Health Care - AI Configuration Module (Day 3).

Centralizes all AI/LLM hyperparameters, timeouts, and retry policies.
All values are tuned for clinical safety and the <4s response NFR.
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
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"      # was llama3-8b-8192
    GROQ_MAIN_MODEL: str = "llama-3.3-70b-versatile"  # was llama3-70b-8192

    # ------------------------------------------------------------------
    # Generation Hyperparameters (Clinical Rationale)
    # ------------------------------------------------------------------
    # Temperature 0.3: Low enough to reduce hallucination and keep
    # outputs clinically conservative, yet high enough to avoid
    # repetitive robotic phrasing that could alarm patients.
    TEMPERATURE: float = 0.3

    # Max tokens 512: Caps response length to guarantee the Groq call
    # + JSON parsing + DB persistence completes within the 3.5s hard
    # timeout, leaving 0.5s buffer before the 4s NFR ceiling.
    MAX_TOKENS: int = 512

    # ------------------------------------------------------------------
    # Timeout & Retry Policy
    # ------------------------------------------------------------------
    # Hard timeout of 3.5s ensures we always have time to fall back
    # gracefully before the user perceives a delay.
    GROQ_TIMEOUT_SECONDS: float = 3.5

    # 3 retries with exponential backoff (1s, 2s, 4s) — capped so total
    # worst-case latency stays bounded. We abort early if the hard
    # timeout is breached.
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE_SECONDS: float = 1.0

    # ------------------------------------------------------------------
    # Structured Output
    # ------------------------------------------------------------------
    # Force JSON mode on Groq so we can reliably extract metadata
    # (specialist, severity) without fragile regex on free text.
    RESPONSE_FORMAT: str = "json"

    # ------------------------------------------------------------------
    # Conversation Memory (Day 3)
    # ------------------------------------------------------------------
    # Limit to 10 messages to balance context richness with token budget
    # and keep memory load time under the 3.5s total latency budget.
    # 10 turns ≈ 20 messages (human + ai) ≈ 200-300 tokens typical.
    MEMORY_MESSAGE_LIMIT: int = 10

    # Max tokens allocated for conversation history.
    # Leaves 60% of the 512-token context window for the current response
    # generation = ~307 tokens for history.
    MEMORY_MAX_TOKENS: int = 307

    # Enable tiktoken for accurate token counting.
    # Set to false on Windows if tiktoken fails to install (Rust extension).
    TIKTOKEN_ENABLED: bool = True

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
        "memory_limit=%d, memory_tokens=%d",
        ai_settings.GROQ_MAIN_MODEL,
        ai_settings.TEMPERATURE,
        ai_settings.MAX_TOKENS,
        ai_settings.GROQ_TIMEOUT_SECONDS,
        ai_settings.MEMORY_MESSAGE_LIMIT,
        ai_settings.MEMORY_MAX_TOKENS,
    )
    return ai_settings


ai_settings = get_ai_settings()