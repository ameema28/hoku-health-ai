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
    GROQ_FAST_MODEL: str = "llama3-8b-8192"
    GROQ_MAIN_MODEL: str = "llama3-70b-8192"

    # Embeddings (stubbed for future RAG pipeline)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Day 5: RAG pipeline (pgvector on Postgres; falls back to in-Python
    # cosine similarity on SQLite -- see app/ai/rag.py)
    VECTOR_DIMENSION: int = 384
    RAG_SIMILARITY_THRESHOLD: float = 0.75
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
