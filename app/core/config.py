"""
Hoku Health Care - Core Configuration Module.

Loads and validates application settings from environment variables
using Pydantic Settings for type safety and runtime validation.

Day 10: adds observability, rate-limiting, and CORS settings. All new
fields carry safe defaults so existing deployments and the 215-test
suite are unaffected when the variables are absent.
"""

import logging
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/hoku_health"

    # Security
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Groq AI
    GROQ_API_KEY: str = ""
    # Day 8.1: aligned with app/ai/config.py AISettings, which is the module
    # the chatbot actually reads. Both read the same case-insensitive env
    # vars; the legacy llama3-*-8192 names are decommissioned on Groq.
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAIN_MODEL: str = "llama-3.3-70b-versatile"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Day 5: RAG pipeline (pgvector on Postgres; in-Python cosine on SQLite)
    VECTOR_DIMENSION: int = 384
    # Day 8.1: lowered 0.75 -> 0.35 to match all-MiniLM-L6-v2 score bands.
    RAG_SIMILARITY_THRESHOLD: float = 0.35
    RAG_TOP_K: int = 3
    COLLECTION_NAME: str = "hoku_health_faqs"

    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # ------------------------------------------------------------------
    # Day 10: Observability
    # ------------------------------------------------------------------
    #: "json" for structured logs (production) or "plain" for local dev.
    LOG_FORMAT: str = "json"
    #: Root log level name.
    LOG_LEVEL: str = "INFO"
    #: Master switch for the /metrics Prometheus endpoint.
    METRICS_ENABLED: bool = True

    # ------------------------------------------------------------------
    # Day 10: Rate limiting (POST /api/ai/chat)
    # ------------------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 5

    # ------------------------------------------------------------------
    # Day 10: CORS (Vercel frontend + local dev)
    # ------------------------------------------------------------------
    #: Comma-separated allowed origins. Parsed by ``cors_origins``.
    CORS_ALLOW_ORIGINS: str = (
        "http://localhost:5173,http://localhost:3000,"
        "https://hoku-health-web.vercel.app"
    )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def cors_origins(self) -> List[str]:
        """
        Parse ``CORS_ALLOW_ORIGINS`` into a clean list.

        Returns:
            List[str]: Trimmed, non-empty origin strings. A lone ``"*"``
            is passed through so wildcard CORS still works in dev.
        """
        raw = (self.CORS_ALLOW_ORIGINS or "").strip()
        if raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_settings() -> Settings:
    """Factory function to create settings instance."""
    return Settings()


def configure_logging() -> None:
    """
    Configure standard library logging for the application.

    Retained for backward compatibility with Day 0-9 callers. New code
    should prefer :func:`app.core.logging.configure_structured_logging`,
    which this function delegates to when structured logging is selected.
    """
    settings_obj = get_settings()
    if settings_obj.LOG_FORMAT.lower() == "json":
        # Import locally to avoid a circular import at module load time.
        from app.core.logging import configure_structured_logging

        configure_structured_logging()
        return

    log_level = logging.DEBUG if settings_obj.DEBUG else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


settings = get_settings()