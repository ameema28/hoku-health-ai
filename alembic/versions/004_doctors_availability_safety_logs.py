"""Add doctors, doctor_availability and safety_logs tables

These tables were previously created only by ``Base.metadata.create_all``
(old init_db.py), so ``alembic upgrade head`` produced a schema without them.
This revision mirrors the models in app/models/ exactly; the migration test
(tests/test_migrations.py) fails if the two drift apart.

Tables or indexes that already exist (databases built by create_all) are
adopted instead of recreated, and each adoption is logged.

Revision ID: 004
Revises: 003
Create Date: 2026-11-08 10:00:00
"""

import logging
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger("alembic.migration")

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _ensure_index(name: str, table: str, columns: list) -> None:
    existing = {ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes(table)}
    if name not in existing:
        op.create_index(name, table, columns, unique=False)


def upgrade() -> None:
    # chat_history.id is declared index=True on the model; 001 never created it.
    _ensure_index("ix_chat_history_id", "chat_history", ["id"])

    if _has_table("doctors"):
        logger.info("doctors already exists - adopting it, not recreating")
    else:
        op.create_table(
            "doctors",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("specialty", sa.String(length=100), nullable=False),
            sa.Column("experience_years", sa.Integer(), nullable=False),
            sa.Column("is_available", sa.Boolean(), nullable=False),
            sa.Column("license_number", sa.String(length=100), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index("ix_doctors_id", "doctors", ["id"])
    _ensure_index("idx_doctors_specialty", "doctors", ["specialty"])
    _ensure_index("idx_doctors_available", "doctors", ["is_available"])

    if _has_table("doctor_availability"):
        logger.info("doctor_availability already exists - adopting it, not recreating")
    else:
        op.create_table(
            "doctor_availability",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("doctor_id", sa.Integer(), nullable=False),
            sa.Column("day_of_week", sa.Integer(), nullable=False),
            sa.Column("start_time", sa.String(length=10), nullable=False),
            sa.Column("end_time", sa.String(length=10), nullable=False),
            sa.Column("is_booked", sa.Boolean(), nullable=False),
            sa.ForeignKeyConstraint(["doctor_id"], ["doctors.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index("ix_doctor_availability_id", "doctor_availability", ["id"])

    if _has_table("safety_logs"):
        logger.info("safety_logs already exists - adopting it, not recreating")
    else:
        op.create_table(
            "safety_logs",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column("ai_response", sa.Text(), nullable=False),
            sa.Column("violation_type", sa.String(length=50), nullable=False),
            sa.Column("severity", sa.String(length=20), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
    _ensure_index("ix_safety_logs_id", "safety_logs", ["id"])
    _ensure_index("ix_safety_logs_user_id", "safety_logs", ["user_id"])
    _ensure_index("ix_safety_logs_violation_type", "safety_logs", ["violation_type"])
    _ensure_index("ix_safety_logs_severity", "safety_logs", ["severity"])
    _ensure_index("ix_safety_logs_created_at", "safety_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("safety_logs")
    op.drop_table("doctor_availability")
    op.drop_table("doctors")
    op.drop_index("ix_chat_history_id", table_name="chat_history")
