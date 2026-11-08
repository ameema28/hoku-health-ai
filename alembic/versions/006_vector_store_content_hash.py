"""Add vector_store.content_hash with a unique index (idempotent FAQ seeding)

Before the index is created, existing rows get their hash (sha256 of the
stripped FAQ question from doc_metadata) and exact duplicate questions are
deleted, keeping the lowest id. Rows without a question get no hash.

Revision ID: 006
Revises: 005
Create Date: 2026-11-08 12:00:00
"""

import hashlib
import json
import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger("alembic.migration")

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX = "ux_vector_store_content_hash"


def _question(raw) -> str:
    meta = json.loads(raw) if isinstance(raw, str) else (raw or {})
    return str(meta.get("question") or "").strip()


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {c["name"] for c in inspector.get_columns("vector_store")}
    if "content_hash" not in columns:
        op.add_column("vector_store", sa.Column("content_hash", sa.String(length=64), nullable=True))

    table = sa.table(
        "vector_store",
        sa.column("id", sa.Integer),
        sa.column("doc_metadata", sa.JSON),
        sa.column("content_hash", sa.String),
    )
    rows = bind.execute(sa.select(table.c.id, table.c.doc_metadata).order_by(table.c.id)).all()
    seen = set()
    duplicates = []
    for row_id, meta in rows:
        question = _question(meta)
        if not question:
            continue
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicates.append(row_id)
            continue
        seen.add(digest)
        bind.execute(table.update().where(table.c.id == row_id).values(content_hash=digest))
    if duplicates:
        bind.execute(table.delete().where(table.c.id.in_(duplicates)))
        logger.info("Removed %d duplicate FAQ rows from vector_store", len(duplicates))

    existing = {ix["name"] for ix in sa.inspect(bind).get_indexes("vector_store")}
    if INDEX not in existing:
        op.create_index(INDEX, "vector_store", ["content_hash"], unique=True)


def downgrade() -> None:
    op.drop_index(INDEX, table_name="vector_store")
    with op.batch_alter_table("vector_store") as batch:
        batch.drop_column("content_hash")
