"""Add vector_store table for Day 5 RAG pipeline

On PostgreSQL with the pgvector extension available: enables pgvector,
creates the vector_store table with a native `vector` column, and adds
an HNSW index for fast approximate nearest-neighbor search.

On SQLite (and any other non-Postgres backend): creates a plain table with
the embedding stored as JSON. HokuRAG (app/ai/rag.py) detects the dialect
at runtime and uses in-Python cosine similarity there.

On PostgreSQL the app always maps the column as `vector` (see
app/models/vector_store.py), so a JSON column would only break retrieval
later. The migration therefore stops with a clear error when pgvector is
not installed on the server, or when the migrating role may not enable it
(pgvector is not a trusted extension: it needs a superuser or, on hosted
Postgres such as Neon, the database owner). An extension that already
exists is accepted, so an administrator can run
`CREATE EXTENSION IF NOT EXISTS vector;` once and then migrate as the app.

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
        if not _pgvector_extension_available(bind):
            raise RuntimeError(
                "The pgvector extension is not installed on this PostgreSQL "
                "server. Use a server with pgvector (e.g. the "
                "pgvector/pgvector image or Neon) and run the migration again."
            )
        try:
            # SAVEPOINT: a failed CREATE EXTENSION must not leave the
            # migration transaction aborted.
            with bind.begin_nested():
                bind.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as exc:
            raise RuntimeError(
                "Could not enable the pgvector extension "
                f"({type(exc).__name__}). Connect as the database owner or a "
                "superuser, run: CREATE EXTENSION IF NOT EXISTS vector; "
                "then run the migration again."
            ) from None
        use_pgvector = True
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
    # An existing table (built by the old create_all path) is adopted as-is.
    inspector = sa.inspect(bind)
    if inspector.has_table("vector_store"):
        logger.info("vector_store already exists - adopting it, not recreating")
    else:
        op.create_table(
            "vector_store",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("content", sa.String(), nullable=False),
            sa.Column("embedding", sa.JSON(), nullable=True),
            sa.Column("doc_metadata", sa.JSON(), nullable=False),
            sa.Column("category", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    existing = {ix["name"] for ix in sa.inspect(bind).get_indexes("vector_store")}
    if "ix_vector_store_category" not in existing:
        op.create_index(
            "ix_vector_store_category", "vector_store", ["category"], unique=False
        )

    if use_pgvector:
        udt = bind.execute(
            sa.text(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name = 'vector_store' AND column_name = 'embedding'"
            )
        ).scalar()
        if udt != "vector":
            op.execute(
                f"ALTER TABLE vector_store ALTER COLUMN embedding TYPE vector({VECTOR_DIMENSION}) "
                f"USING embedding::text::vector({VECTOR_DIMENSION})"
            )
        # HNSW index for fast approximate cosine-similarity search.
        # Requires pgvector >= 0.5.0.
        op.execute(
            "CREATE INDEX IF NOT EXISTS ix_vector_store_embedding_hnsw "
            "ON vector_store USING hnsw (embedding vector_cosine_ops)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_vector_store_embedding_hnsw")
    op.drop_index("ix_vector_store_category", table_name="vector_store")
    op.drop_table("vector_store")
