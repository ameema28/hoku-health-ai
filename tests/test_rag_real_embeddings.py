"""The SQLite (in-Python cosine) path with the real MiniLM model gives the same top-1 as pgvector."""

import pytest

from app.ai.embeddings import EmbeddingManager
from tests.rag_expectations import EXPECTED_TOP1


@pytest.fixture(scope="module")
def real_model():
    manager = EmbeddingManager()
    if not any(manager.get_embedding("probe")):
        pytest.skip("embedding model not available offline")
    return manager


@pytest.mark.parametrize("query,expected", sorted(EXPECTED_TOP1.items()))
def test_sqlite_path_top1_matches_pgvector_expectation(db, real_model, query, expected):
    from app.ai.rag import HokuRAG
    from app.scripts.seed_faqs import FAQS

    rag = HokuRAG(db=db)
    assert rag._is_postgres is False
    rag.embedding_manager = real_model
    rag.upsert_faq_documents(FAQS)
    results = rag.similarity_search(query, k=3)
    assert results[0].metadata["question"] == expected
