"""
Hoku Health Care - RAG Pipeline Unit Tests (Day 5).

All pgvector-dependent tests are mocked, since SQLite (used for local
test runs) doesn't support the pgvector extension. These tests verify
the embedding manager and HokuRAG's Python-level logic (thresholding,
context building) rather than exercising a live PostgreSQL+pgvector
connection.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from app.ai.embeddings import EmbeddingManager
from app.ai.rag import HokuRAG


# ---------------------------------------------------------------------------
# Embedding manager tests
# ---------------------------------------------------------------------------
def test_get_embedding_returns_384_dim_vector():
    """get_embedding should return a 384-dimensional vector."""
    manager = EmbeddingManager()
    vector = manager.get_embedding("Do you offer home nursing in Lahore?")
    assert isinstance(vector, list)
    assert len(vector) == 384
    assert all(isinstance(x, float) for x in vector)


def test_batch_embed_returns_correct_shape():
    """batch_embed should return one 384-dim vector per input text."""
    manager = EmbeddingManager()
    texts = ["What services do you offer?", "How do I book a nurse?", "Emergency numbers?"]
    vectors = manager.batch_embed(texts)
    assert len(vectors) == len(texts)
    for vector in vectors:
        assert len(vector) == 384


def test_get_embedding_empty_text_returns_zero_vector():
    """Empty input should return a safe zero-vector, not raise."""
    manager = EmbeddingManager()
    vector = manager.get_embedding("")
    assert vector == [0.0] * 384


# ---------------------------------------------------------------------------
# HokuRAG tests (pgvector interactions mocked)
# ---------------------------------------------------------------------------
def _make_document(question: str, answer: str, category: str, score: float) -> Document:
    return Document(
        page_content=f"Q: {question}\nA: {answer}",
        metadata={"question": question, "answer": answer, "category": category, "score": score},
    )


def test_similarity_search_returns_document_list():
    """similarity_search should return a list of Document objects."""
    rag = HokuRAG.__new__(HokuRAG)  # bypass __init__ (avoids real DB/model load)
    mock_results = [
        _make_document("What services do you offer?", "Home nursing, physio...", "services", 0.91),
        _make_document("How do I book a nurse?", "Use the app...", "booking", 0.83),
    ]
    with patch.object(HokuRAG, "similarity_search", return_value=mock_results):
        results = rag.similarity_search("What does Hoku Health Care offer?", k=3)

    assert isinstance(results, list)
    assert all(isinstance(doc, Document) for doc in results)
    assert results[0].metadata["score"] == 0.91


def test_build_context_with_relevant_query_returns_non_empty_string():
    """A query with a strong FAQ match should produce a non-empty context string."""
    rag = HokuRAG.__new__(HokuRAG)
    rag.top_k = 3
    rag.similarity_threshold = 0.75

    mock_results = [
        _make_document(
            "Do you provide home nursing care in Lahore?",
            "Yes, Hoku Health Care provides licensed home nursing in Lahore.",
            "services",
            0.88,
        ),
    ]

    with patch.object(HokuRAG, "similarity_search", return_value=mock_results):
        context = rag.build_context("home nursing in Lahore")

    assert context != ""
    assert "Lahore" in context


def test_build_context_with_irrelevant_query_returns_empty_string():
    """A query with only weak FAQ matches (below threshold) should return ''."""
    rag = HokuRAG.__new__(HokuRAG)
    rag.top_k = 3
    rag.similarity_threshold = 0.75

    mock_results = [
        _make_document(
            "What are Hoku Health Care's operating hours?",
            "Teleconsultations are available 24/7.",
            "general",
            0.42,  # below threshold
        ),
    ]

    with patch.object(HokuRAG, "similarity_search", return_value=mock_results):
        context = rag.build_context("what's the weather like today")

    assert context == ""


def test_build_context_with_no_results_returns_empty_string():
    """If similarity_search finds nothing at all, build_context should return ''."""
    rag = HokuRAG.__new__(HokuRAG)
    rag.top_k = 3
    rag.similarity_threshold = 0.75

    with patch.object(HokuRAG, "similarity_search", return_value=[]):
        context = rag.build_context("completely unrelated query")

    assert context == ""
