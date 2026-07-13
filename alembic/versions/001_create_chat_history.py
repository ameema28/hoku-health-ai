"""
Create chat_history table

Revision ID: 001
Revises:
Create Date: 2026-07-13 15:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Create the chat_history table with required columns and indexes.

    Indexes:
        - ix_chat_history_user_id: speeds up per-user lookups.
        - ix_chat_history_created_at: speeds up time-range queries.
    """
    op.create_table(
        "chat_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("ai_response", sa.Text(), nullable=True),
        sa.Column("intent", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        # sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
    )
    op.create_index(
        "ix_chat_history_user_id",
        "chat_history",
        ["user_id"],
    )
    op.create_index(
        "ix_chat_history_created_at",
        "chat_history",
        ["created_at"],
    )


def downgrade() -> None:
    """Drop the chat_history table and its indexes."""
    op.drop_index("ix_chat_history_created_at", table_name="chat_history")
    op.drop_index("ix_chat_history_user_id", table_name="chat_history")
    op.drop_table("chat_history")
