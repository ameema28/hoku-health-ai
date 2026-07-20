"""Add vector_store table for Day 5 RAG pipeline

On PostgreSQL with the pgvector extension available: enables pgvector,
creates the vector_store table with a native `vector` column, and adds
an HNSW index for fast approximate nearest-neighbor search.

On SQLite (and any other non-Postgres backend, or Postgres without
pgvector installed): creates a plain table with the embedding stored as
JSON instead. HokuRAG (app/ai/rag.py) already detects the active dialect
at runtime and falls back to in-Python cosine similarity in that case,
so this is safe to run as-is against a SQLite dev database.

Revision ID: 003
Revises: 002
Create Date: 2026-07-18 10:00:00
"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger("alembic.migration")

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

VECTOR_DIMENSION = 384  # sentence-transformers/all-MiniLM-L6-v2


def _pgvector_extension_available(bind) -> bool:
    """Best-effort check: does this Postgres server already have pgvector?"""
    try:
        result = bind.execute(
            sa.text("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        )
        return result.scalar() is not None
    except Exception:
        return False


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"

    use_pgvector = False
    if is_postgres:
        if _pgvector_extension_available(bind):
            try:
                op.execute("CREATE EXTENSION IF NOT EXISTS vector")
                use_pgvector = True
            except Exception as exc:
                logger.warning(
                    "Could not enable pgvector extension (may require superuser "
                    "privileges): %s. Falling back to a JSON embedding column.",
                    exc,
                )
        else:
            logger.warning(
                "pgvector extension not available on this Postgres server. "
                "Falling back to a JSON embedding column -- install pgvector "
                "and re-run this migration to get native vector search."
            )
    else:
        logger.info(
            "Non-Postgres dialect ('%s') detected -- creating vector_store with "
            "a JSON embedding column. HokuRAG falls back to in-Python cosine "
            "similarity on this backend.",
            bind.dialect.name,
        )

    # Portable base table: JSON embedding column works on every backend
    # (SQLite, Postgres-without-pgvector, etc). We upgrade the column type
    # to native `vector` below only when pgvector is actually available.
    op.create_table(
        "vector_store",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("doc_metadata", sa.JSON(), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_index("ix_vector_store_category", "vector_store", ["category"], unique=False)

    if use_pgvector:
        op.execute(
            f"ALTER TABLE vector_store ALTER COLUMN embedding TYPE vector({VECTOR_DIMENSION}) "
            f"USING embedding::text::vector({VECTOR_DIMENSION})"
        )
        # HNSW index for fast approximate cosine-similarity search.
        # Requires pgvector >= 0.5.0.
        op.execute(
            "CREATE INDEX ix_vector_store_embedding_hnsw "
            "ON vector_store USING hnsw (embedding vector_cosine_ops)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_vector_store_embedding_hnsw")
    op.drop_index("ix_vector_store_category", table_name="vector_store")
    op.drop_table("vector_store")
