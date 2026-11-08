"""pgvector similarity search, unmocked: real Postgres, real migrations, real MiniLM."""

import pytest
from sqlalchemy import event, func, select, text

from app.models.vector_store import VectorStore
from tests.pgvector.conftest import IS_POSTGRES
from tests.rag_expectations import EXPECTED_TOP1

pytestmark = [
    pytest.mark.pgvector,
    pytest.mark.skipif(not IS_POSTGRES, reason="needs DATABASE_URL=postgresql://..."),
]


def test_migrations_create_vector_column_and_hnsw_index(pg_engine):
    with pg_engine.connect() as conn:
        udt = conn.execute(
            text(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name = 'vector_store' AND column_name = 'embedding'"
            )
        ).scalar()
        hnsw = conn.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE indexname = 'ix_vector_store_embedding_hnsw'"
            )
        ).scalar()
        extension = conn.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        ).scalar()
    assert udt == "vector"
    assert extension == 1
    assert hnsw is not None and "hnsw" in hnsw and "vector_cosine_ops" in hnsw


def test_models_match_migrated_postgres_schema(pg_engine):
    """Same parity check as tests/test_migrations.py, on the real backend."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    import app.models  # noqa: F401
    from app.core.database import Base

    with pg_engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(ctx, Base.metadata) == []


def test_seeded_twenty_faqs(seeded_session):
    count = seeded_session.scalar(select(func.count()).select_from(VectorStore))
    assert count == 20


def test_search_uses_pgvector_operator(seeded_session, pg_engine):
    from app.ai.rag import HokuRAG

    statements = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)

    event.listen(pg_engine, "before_cursor_execute", _capture)
    try:
        results = HokuRAG(db=seeded_session).similarity_search(
            "what are your opening hours", k=3
        )
    finally:
        event.remove(pg_engine, "before_cursor_execute", _capture)

    assert len(results) == 3
    assert any("<=>" in s for s in statements), statements


@pytest.mark.parametrize("query,expected", sorted(EXPECTED_TOP1.items()))
def test_paraphrased_query_top1(seeded_session, query, expected):
    from app.ai.rag import HokuRAG

    results = HokuRAG(db=seeded_session).similarity_search(query, k=3)
    assert results, "pgvector search returned nothing (F3 regression)"
    assert results[0].metadata["question"] == expected
    scores = [doc.metadata["score"] for doc in results]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in scores)


def test_build_context_grounds_on_pgvector(seeded_session):
    from app.ai.rag import HokuRAG

    context = HokuRAG(db=seeded_session).build_context(
        "nursing care at home after an operation", threshold=0.35
    )
    assert "post-surgical" in context
