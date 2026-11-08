"""Add intent index to chat_history

Revision ID: 002
Revises: 001
Create Date: 2026-07-17 15:58:00

Skips the index if it already exists (databases built by create_all).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    existing = {ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes("chat_history")}
    if "ix_chat_history_intent" not in existing:
        op.create_index("ix_chat_history_intent", "chat_history", ["intent"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_chat_history_intent", table_name="chat_history")
