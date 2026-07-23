"""
Hoku Health Care - Core Configuration Module.

Loads and validates application settings from environment variables
using Pydantic Settings for type safety and runtime validation.
"""

import logging
from typing import Optional

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
    # the chatbot actually reads. These two files previously declared
    # DIFFERENT default models (llama3-8b-8192 / llama3-70b-8192 here vs
    # llama-3.1-8b-instant / llama-3.3-70b-versatile there). Because both read
    # the same case-insensitive env vars, whichever .env value was set
    # silently overrode one of them — and the legacy llama3-*-8192 names are
    # decommissioned on Groq, so any code path reading these would 404.
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAIN_MODEL: str = "llama-3.3-70b-versatile"

    # Embeddings (stubbed for future RAG pipeline)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Day 5: RAG pipeline (pgvector on Postgres; falls back to in-Python
    # cosine similarity on SQLite -- see app/ai/rag.py)
    VECTOR_DIMENSION: int = 384

    # Day 8.1: lowered 0.75 -> 0.35.
    #
    # all-MiniLM-L6-v2 with normalize_embeddings=True produces cosine
    # similarities in roughly these bands for short FAQ text:
    #     0.00-0.25  unrelated
    #     0.25-0.35  weakly related
    #     0.35-0.55  on-topic paraphrase   <- what we want to retrieve
    #     0.55-1.00  near-duplicate wording
    #
    # A 0.75 gate is effectively "near-verbatim match only", so build_context
    # returned "" on essentially every real query. Observed in production
    # logs: top score 0.233 vs threshold 0.750 -> RAG never fired, and the
    # entire Day 5 deliverable (rag_chat_prompt_template, the {faq_context}
    # slot) was unreachable code.
    #
    # 0.35 sits above the unrelated band, so a genuinely irrelevant FAQ is
    # still rejected rather than grounding a clinical reply on noise.
    # Tune per corpus: raise it if replies cite unrelated FAQs.
    RAG_SIMILARITY_THRESHOLD: float = 0.35
    RAG_TOP_K: int = 3
    COLLECTION_NAME: str = "hoku_health_faqs"

    # Application
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"


def get_settings() -> Settings:
    """Factory function to create settings instance."""
    return Settings()


def configure_logging() -> None:
    """Configure standard library logging for the application."""
    log_level = logging.DEBUG if get_settings().DEBUG else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


settings = get_settings()