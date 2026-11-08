"""Role-based access: admin-only routes, patient routes, /metrics scrape token, FAQ seeding."""

import hashlib
from unittest.mock import patch

import pytest
from sqlalchemy import func, select

from app.core.config import settings
from app.models.vector_store import VectorStore
from tests.auth_utils import add_user, bearer

PATIENT_ID, ADMIN_ID = 1, 2

ADMIN_ROUTES = [
    ("post", "/api/ai/rag/seed"),
    ("get", "/api/ai/rag/search?q=opening+hours"),
    ("get", "/api/ai/monitoring/metrics"),
]
PATIENT_ROUTES = [
    ("get", "/api/ai/chat/history"),
    ("get", "/api/ai/doctors?specialty=Cardiologist"),
    ("get", "/api/ai/doctors/1/availability"),
]


def _fake_embedding(text):
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255.0 + 0.01 for b in digest[:16]] * 24  # 384 dims, non-zero


@pytest.fixture(autouse=True)
def _fake_embeddings():
    with patch(
        "app.ai.rag.EmbeddingManager.get_embedding", side_effect=_fake_embedding
    ):
        yield


@pytest.fixture
def users(db):
    add_user(db, user_id=PATIENT_ID, role="patient")
    add_user(db, user_id=ADMIN_ID, role="admin")
    return {
        "patient": {"Authorization": bearer(PATIENT_ID, "patient")},
        "admin": {"Authorization": bearer(ADMIN_ID, "admin")},
    }


def _call(client, method, path, headers=None):
    return getattr(client, method)(path, headers=headers or {})


@pytest.mark.parametrize("method,path", ADMIN_ROUTES)
class TestAdminRoutes:
    def test_no_token_401(self, client, users, method, path):
        assert _call(client, method, path).status_code == 401

    def test_invalid_token_401(self, client, users, method, path):
        bad = {"Authorization": "Bearer not-a-token"}
        assert _call(client, method, path, bad).status_code == 401

    def test_patient_403(self, client, users, method, path):
        assert _call(client, method, path, users["patient"]).status_code == 403

    def test_admin_200(self, client, users, method, path):
        assert _call(client, method, path, users["admin"]).status_code == 200


@pytest.mark.parametrize("method,path", PATIENT_ROUTES)
class TestPatientRoutes:
    def test_no_token_401(self, client, users, method, path):
        assert _call(client, method, path).status_code == 401

    @pytest.mark.parametrize("role", ["patient", "admin"])
    def test_both_roles_200(self, client, users, method, path, role):
        assert _call(client, method, path, users[role]).status_code == 200


class TestChatRoute:
    def test_no_token_401(self, client, users):
        resp = client.post(
            "/api/ai/chat", json={"message": "hello", "userId": PATIENT_ID}
        )
        assert resp.status_code == 401

    @pytest.mark.parametrize(
        "role,user_id", [("patient", PATIENT_ID), ("admin", ADMIN_ID)]
    )
    def test_both_roles_allowed(self, client, users, role, user_id):
        # No GROQ key in tests: the chatbot answers with its safe fallback.
        resp = client.post(
            "/api/ai/chat",
            json={"message": "hello", "userId": user_id},
            headers=users[role],
        )
        assert resp.status_code == 200


class TestSeedIdempotency:
    def test_second_seed_adds_nothing(self, client, db, users):
        first = client.post("/api/ai/rag/seed", headers=users["admin"]).json()
        second = client.post("/api/ai/rag/seed", headers=users["admin"]).json()
        assert (first["documents_added"], first["documents_skipped"]) == (20, 0)
        assert (second["documents_added"], second["documents_skipped"]) == (0, 20)
        assert db.scalar(select(func.count()).select_from(VectorStore)) == 20

    def test_duplicates_within_one_batch_are_skipped(self, db):
        from app.ai.rag import HokuRAG

        faq = {"question": "Same question?", "answer": "A", "category": "general"}
        added, skipped = HokuRAG(db=db).upsert_faq_documents([faq, dict(faq)])
        assert (added, skipped) == (1, 1)


class TestMetricsEndpoint:
    TOKEN = "scrape-token-for-tests-0123456789"

    @pytest.fixture
    def cfg(self, monkeypatch):
        def _set(env="test", token="", enabled=True):
            monkeypatch.setattr(settings, "ENVIRONMENT", env)
            monkeypatch.setattr(settings, "METRICS_TOKEN", token)
            monkeypatch.setattr(settings, "METRICS_ENABLED", enabled)

        return _set

    @pytest.mark.parametrize(
        "env,expected",
        [("test", 200), ("development", 200), ("production", 404), ("staging", 404)],
    )
    def test_token_unset(self, client, cfg, env, expected):
        cfg(env=env)
        assert client.get("/metrics").status_code == expected

    @pytest.mark.parametrize("env", ["test", "production"])
    @pytest.mark.parametrize(
        "header,expected",
        [
            (None, 401),
            ("Bearer wrong-token", 401),
            (TOKEN, 401),  # missing "Bearer " prefix
            (f"Bearer {TOKEN}", 200),
        ],
    )
    def test_token_set(self, client, cfg, env, header, expected):
        cfg(env=env, token=self.TOKEN)
        headers = {"Authorization": header} if header else {}
        resp = client.get("/metrics", headers=headers)
        assert resp.status_code == expected
        if expected == 200:
            assert "hoku_chatbot_requests_total" in resp.text

    def test_disabled_404_even_with_token(self, client, cfg):
        cfg(token=self.TOKEN, enabled=False)
        assert (
            client.get(
                "/metrics", headers={"Authorization": f"Bearer {self.TOKEN}"}
            ).status_code
            == 404
        )
