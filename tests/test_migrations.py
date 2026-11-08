"""Alembic migrations: full upgrade/downgrade, model parity, adoption and seeding."""

import hashlib
from unittest.mock import patch

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401  (registers all tables)
import init_db
from app.core.database import Base
from app.models import Doctor, DoctorAvailability, VectorStore

APP_TABLES = {
    "chat_history",
    "vector_store",
    "doctors",
    "doctor_availability",
    "safety_logs",
}


@pytest.fixture
def mig_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'mig.db'}")
    yield engine
    engine.dispose()


def _run(engine, fn, revision):
    cfg = init_db.alembic_config()
    cfg.attributes["configure_logger"] = False
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        fn(cfg, revision)


def _diff(engine):
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        return compare_metadata(ctx, Base.metadata)


def _fake_embedding(text):
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 for b in digest[:16]] * 24  # 384 dims, non-zero


def test_upgrade_head_matches_models(mig_engine):
    _run(mig_engine, command.upgrade, "head")
    assert APP_TABLES <= set(inspect(mig_engine).get_table_names())
    assert _diff(mig_engine) == []


def test_downgrade_base_removes_everything(mig_engine):
    _run(mig_engine, command.upgrade, "head")
    _run(mig_engine, command.downgrade, "base")
    assert not APP_TABLES & set(inspect(mig_engine).get_table_names())


def test_upgrade_adopts_database_built_by_create_all(mig_engine):
    """Old init_db.py used create_all; upgrade head must not fail on it."""
    Base.metadata.create_all(bind=mig_engine)
    _run(mig_engine, command.upgrade, "head")
    assert _diff(mig_engine) == []


def test_seed_doctors_is_idempotent(mig_engine):
    from app.scripts.seed_doctors import SAMPLE_DOCTORS, WEEKLY_SLOTS, seed_doctors

    _run(mig_engine, command.upgrade, "head")
    session = sessionmaker(bind=mig_engine)()
    try:
        first = seed_doctors(session)
        second = seed_doctors(session)
        n_doctors = session.scalar(select(func.count()).select_from(Doctor))
        n_slots = session.scalar(select(func.count()).select_from(DoctorAvailability))
    finally:
        session.close()

    assert first["doctors_added"] == len(SAMPLE_DOCTORS) == 7
    assert second == {"doctors_added": 0, "doctors_skipped": 7, "slots_added": 0}
    assert n_doctors == 7
    assert n_slots == 7 * len(WEEKLY_SLOTS)


def test_faq_seed_runs_against_migrated_db(mig_engine):
    from app.ai.rag import HokuRAG
    from app.scripts.seed_faqs import FAQS

    _run(mig_engine, command.upgrade, "head")
    session = sessionmaker(bind=mig_engine)()
    try:
        with patch(
            "app.ai.rag.EmbeddingManager.get_embedding", side_effect=_fake_embedding
        ):
            rag = HokuRAG(db=session)
            rag.create_vector_store()
            added = rag.add_faq_documents(FAQS)
        count = session.scalar(select(func.count()).select_from(VectorStore))
    finally:
        session.close()
    assert added == len(FAQS) == 20
    assert count == 20


def test_create_vector_store_requires_migrations(mig_engine):
    from app.ai.rag import HokuRAG

    session = sessionmaker(bind=mig_engine)()
    try:
        with pytest.raises(RuntimeError, match="Run migrations first"):
            HokuRAG(db=session).create_vector_store()
    finally:
        session.close()
    assert "vector_store" not in inspect(mig_engine).get_table_names()


def test_init_db_main_migrates_and_seeds(mig_engine, monkeypatch, capsys):
    url = str(mig_engine.url)
    real_config = init_db.alembic_config

    def _cfg():
        cfg = real_config()
        cfg.set_main_option("sqlalchemy.url", url)
        cfg.attributes["configure_logger"] = False
        return cfg

    monkeypatch.setattr(init_db, "alembic_config", _cfg)
    monkeypatch.setattr("app.core.database.SessionLocal", sessionmaker(bind=mig_engine))
    init_db.main(["--no-faqs"])
    init_db.main(["--no-faqs"])  # second run is a no-op

    session = sessionmaker(bind=mig_engine)()
    try:
        assert session.scalar(select(func.count()).select_from(Doctor)) == 7
    finally:
        session.close()
    assert "Doctors: 0 added, 7 already present." in capsys.readouterr().out
