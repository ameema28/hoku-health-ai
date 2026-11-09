"""
Hoku Health Care - Core Configuration Module.

Loads and validates application settings from environment variables
using Pydantic Settings for type safety and runtime validation.

Day 10: adds observability, rate-limiting, and CORS settings. All new
fields carry safe defaults so existing deployments and the 215-test
suite are unaffected when the variables are absent.
"""

import os
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``.env`` is never read when ENVIRONMENT=test (taken from the real process
#: environment), so a developer's local secrets cannot leak into test runs.
ENV_FILE: Optional[str] = (
    None if os.getenv("ENVIRONMENT", "").lower() == "test" else ".env"
)


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database. PostgreSQL URLs pass straight to psycopg2, so libpq options
    # such as ?sslmode=require (Neon) work unchanged.
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/hoku_health"
    #: SQLAlchemy pool (PostgreSQL). Small values suit a free hosted database
    #: and one Cloud Run instance, e.g. 2 and 3.
    DB_POOL_SIZE: int = Field(default=10, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0)

    # Security. No default key: outside ENVIRONMENT=test the app refuses to
    # start unless SECRET_KEY has 32+ characters (see
    # app.core.security.validate_security_settings). JWTs are always HS256.
    SECRET_KEY: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_ISSUER: str = "hoku-health-ai"
    JWT_AUDIENCE: str = "hoku-health-api"

    # Demo login (POST /api/auth/demo). Off unless explicitly enabled.
    DEMO_LOGIN_ENABLED: bool = False
    DEMO_TOKEN_EXPIRE_MINUTES: int = 15
    #: Per client IP (see TRUST_PROXY). Everyone behind one public IP
    #: (shared Wi-Fi, carrier NAT) shares this limit.
    DEMO_RATE_LIMIT_PER_HOUR: int = 20

    # Client IP for the per-IP limits (app/core/client_ip.py). With
    # TRUST_PROXY=false the socket peer is used and X-Forwarded-For is
    # ignored. With TRUST_PROXY=true the entry TRUST_PROXY_HOPS places from
    # the right of X-Forwarded-For is used (1 = the entry the proxy appended).
    TRUST_PROXY: bool = False
    TRUST_PROXY_HOPS: int = Field(default=1, ge=1)

    # Groq AI
    GROQ_API_KEY: str = ""
    # Model names live in app/ai/config.py (AISettings).

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
    #: Scrape token for /metrics ("Authorization: Bearer <token>"). When
    #: unset, /metrics is open only in ENVIRONMENT test/development and
    #: returns 404 everywhere else.
    METRICS_TOKEN: str = ""

    # ------------------------------------------------------------------
    # Day 10: Rate limiting (POST /api/ai/chat)
    # ------------------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = True
    #: Per authenticated user (each demo session is a new user).
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 10
    #: Per client IP, across all users (proxy-aware, see TRUST_PROXY).
    CHAT_IP_RATE_PER_MINUTE: int = 20

    # ------------------------------------------------------------------
    # CORS: browser origins (other than the API's own) allowed to call it.
    # The demo page at / is same-origin and needs none. Local dev only by
    # default; add a deployed frontend's origin explicitly.
    # ------------------------------------------------------------------
    #: Comma-separated allowed origins. Parsed by ``cors_origins``.
    CORS_ALLOW_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

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


settings = get_settings()
