"""Startup RAG warm-up, the rag_ready flag and the health payload."""

from unittest.mock import MagicMock

import pytest

from app.ai import readiness
from app.ai.embeddings import EmbeddingManager
from app.ai.rag import HokuRAG
from app.scripts.seed_faqs import FAQS


@pytest.fixture(autouse=True)
def _restore_flag():
    before = readiness.rag_ready()
    yield
    readiness.set_rag_ready(before)


class TestHealthPayload:
    def test_ready(self):
        readiness.set_rag_ready(True)
        assert readiness.health_payload() == {
            "status": "ok",
            "service": "Hoku AI Chatbot",
            "rag_ready": True,
        }

    def test_degraded(self):
        readiness.set_rag_ready(False)
        body = readiness.health_payload()
        assert body["status"] == "degraded"
        assert body["rag_ready"] is False

    @pytest.mark.parametrize("ready", [True, False])
    def test_both_routes_return_the_same_payload(self, client, ready):
        readiness.set_rag_ready(ready)
        a = client.get("/health")
        b = client.get("/api/ai/health")
        assert a.status_code == b.status_code == 200
        assert a.json() == b.json() == readiness.health_payload()


class TestCheckRag:
    def test_results_mean_ready(self):
        rag = MagicMock()
        rag.similarity_search.return_value = [object()]
        assert readiness.check_rag(rag) is True
        assert readiness.rag_ready() is True
        rag.similarity_search.assert_called_once_with(readiness.WARM_UP_QUERY, k=1)

    def test_no_results_mean_not_ready(self):
        rag = MagicMock()
        rag.similarity_search.return_value = []
        readiness.set_rag_ready(True)
        assert readiness.check_rag(rag) is False
        assert readiness.rag_ready() is False

    def test_exception_means_not_ready(self):
        rag = MagicMock()
        rag.similarity_search.side_effect = RuntimeError("boom")
        assert readiness.check_rag(rag) is False

    def test_empty_faq_table_is_not_ready(self, db):
        assert readiness.check_rag(HokuRAG(db=db)) is False

    def test_seeded_index_with_real_model_is_ready(self, db):
        rag = HokuRAG(db=db)
        rag.upsert_faq_documents(FAQS)
        assert readiness.check_rag(rag) is True

    def test_zero_vector_fallback_is_not_ready(self, db, monkeypatch):
        rag = HokuRAG(db=db)
        rag.upsert_faq_documents(FAQS)
        monkeypatch.setattr(
            rag.embedding_manager, "get_embedding", lambda text: [0.0] * 384
        )
        assert readiness.check_rag(rag) is False


def test_embedding_model_is_loaded_once_per_process():
    first, second = EmbeddingManager(), EmbeddingManager()
    assert first.model is not None
    assert first.model is second.model


async def test_warm_up_uses_the_serving_chatbot(monkeypatch):
    from app import main
    from app.services import ai_service

    fake_rag = MagicMock()
    fake_rag.similarity_search.return_value = [object()]
    monkeypatch.setattr(ai_service._chatbot, "_rag", fake_rag)
    readiness.set_rag_ready(False)
    await main._warm_up_rag()
    assert readiness.rag_ready() is True
    fake_rag.similarity_search.assert_called_once()


def test_admin_seed_updates_readiness(client, db):
    from app.core.security import require_admin
    from app.main import app

    app.dependency_overrides[require_admin] = lambda: {"id": 1, "role": "admin"}
    readiness.set_rag_ready(False)
    resp = client.post("/api/ai/rag/seed")
    assert resp.status_code == 200
    assert readiness.rag_ready() is True
