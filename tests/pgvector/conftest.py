"""Fixtures for tests that need a real PostgreSQL + pgvector database.

Everything here is skipped unless DATABASE_URL points at PostgreSQL (the
embedding column type is chosen when app.models.vector_store is imported).
The target database's public schema is dropped and rebuilt by migrations.
"""

import pytest
from alembic import command
from sqlalchemy import text

from app.core.config import settings

IS_POSTGRES = settings.DATABASE_URL.startswith("postgresql")


@pytest.fixture(scope="module")
def pg_engine():
    if not IS_POSTGRES:
        pytest.skip("pgvector tests need DATABASE_URL=postgresql://...")
    import init_db
    from app.core.database import engine

    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    cfg = init_db.alembic_config()
    cfg.attributes["configure_logger"] = False
    command.upgrade(cfg, "head")  # 003 must create the extension itself
    yield engine


@pytest.fixture(scope="module")
def seeded_session(pg_engine):
    from app.ai.rag import HokuRAG
    from app.core.database import SessionLocal
    from app.scripts.seed_faqs import FAQS

    session = SessionLocal()
    rag = HokuRAG(db=session)
    if not any(rag.embedding_manager.get_embedding("probe")):
        pytest.skip("embedding model not available")
    rag.create_vector_store()
    rag.upsert_faq_documents(FAQS)
    yield session
    session.close()
