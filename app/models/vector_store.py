"""
Hoku Health Care - Vector Store Model (Day 5).

SQLAlchemy model backing the pgvector-based FAQ collection. This model
is used directly by HokuRAG for similarity search rather than going
through LangChain's PGVector wrapper, which keeps the schema explicit
and lets us index on `category` for filtered searches.

Note: pgvector.sqlalchemy.Vector requires the `pgvector` PostgreSQL
extension (CREATE EXTENSION IF NOT EXISTS vector) -- see
alembic/versions/003_add_vector_store.py. Your environment is
SQLite-only with no pgvector installed, so the `embedding` column below
resolves to a plain JSON column, and app/ai/rag.py automatically falls
back to computing cosine similarity in Python. Everything still works
end-to-end -- it just isn't using pgvector's native ANN index. If you
later move to PostgreSQL and install pgvector, this model switches to
the real `vector` column type automatically (see _USE_PGVECTOR_COLUMN
below), no code changes needed.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import settings
from app.core.database import Base

try:
    from pgvector.sqlalchemy import Vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    # Expected in your environment: pgvector isn't installed at all.
    # This keeps the module importable -- the embedding column below
    # falls back to JSON instead of raising ImportError.
    Vector = None  # type: ignore
    PGVECTOR_AVAILABLE = False

# Use the real pgvector column type only when both the package is
# installed AND we're actually pointed at PostgreSQL. On SQLite (your
# current setup) we fall back to a JSON column so the model -- and
# every module that imports it -- stays importable and testable
# without a live pgvector database.
_USE_PGVECTOR_COLUMN = PGVECTOR_AVAILABLE and settings.DATABASE_URL.startswith("postgresql")


class VectorStore(Base):
    """A single embedded FAQ entry in the Hoku Health Care knowledge base."""

    __tablename__ = "vector_store"
    __table_args__ = (Index("ix_vector_store_category", "category"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    # The FAQ text (question+answer combined) that was embedded.
    content: Mapped[str] = mapped_column(String, nullable=False)

    # 384-dim embedding produced by sentence-transformers/all-MiniLM-L6-v2.
    # JSON on SQLite (your current setup); native `vector` type on Postgres
    # with pgvector installed.
    embedding: Mapped[Optional[list]] = mapped_column(
        Vector(settings.VECTOR_DIMENSION) if _USE_PGVECTOR_COLUMN else JSON,
        nullable=True,
    )

    # JSON metadata: {"question": ..., "answer": ..., "category": ...}
    # Named doc_metadata (not "metadata") because SQLAlchemy's
    # declarative Base reserves the `metadata` attribute name.
    doc_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Denormalized for fast filtered lookups without JSON parsing.
    # Indexed via __table_args__ above (not index=True here, to avoid a
    # duplicate index definition).
    category: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc), nullable=False
    )
