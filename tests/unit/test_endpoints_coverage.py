"""
Hoku Health Care - Day 10 endpoint & CRUD coverage tests.

Exercises the API endpoints and safety-audit CRUD paths that the six
main integration tests don't touch: health, doctor lookup, RAG debug
search, Prometheus /metrics, the monitoring summary, and the
SafetyLog persistence layer. Together these cover large blocks of
app/api/v1/endpoints/ai.py, app/crud/crud_safety.py, and app/crud/
crud_doctor.py without any Groq call - only GETs and direct DB writes.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.database import Base, get_db
from app.core.rate_limit import reset_chat_rate_limiter
from app.main import app

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_endpoints.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

USER_ID = 1


def _token(user_id: int = USER_ID) -> str:
    """Mint a Bearer token for the auth stub."""
    raw = jwt.encode(
        {
            "sub": str(user_id),
            "id": user_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2),
        },
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return f"Bearer {raw}"


@pytest.fixture(scope="function")
def db() -> Iterator[Any]:
    """Fresh schema per test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db: Any) -> Iterator[TestClient]:
    """TestClient with DB override and a reset rate limiter."""
    reset_chat_rate_limiter()

    def _override() -> Iterator[Any]:
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def auth() -> Dict[str, str]:
    """Authorization header."""
    return {"Authorization": _token()}


# ---------------------------------------------------------------------------
# Endpoints that need no Groq call
# ---------------------------------------------------------------------------
class TestUnauthenticatedEndpoints:
    """Health, root, and metrics are open and cheap."""

    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/api/ai/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_root_ok(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "version" in resp.json()

    def test_metrics_prometheus_exposition(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "hoku_chatbot_requests_total" in resp.text


class TestAuthGating:
    """Authenticated endpoints reject missing tokens."""

    def test_chat_history_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ai/chat/history")
        assert resp.status_code == 401

    def test_doctors_requires_auth(self, client: TestClient) -> None:
        resp = client.get("/api/ai/doctors?specialty=Cardiologist")
        assert resp.status_code == 401


class TestDoctorEndpoints:
    """Doctor lookup routes work with an empty DB (return empty lists)."""

    def test_list_doctors_empty(self, client: TestClient, auth: Dict[str, str]) -> None:
        resp = client.get("/api/ai/doctors?specialty=Cardiologist", headers=auth)
        assert resp.status_code == 200
        assert resp.json() == []

    def test_doctor_availability_empty(self, client: TestClient, auth: Dict[str, str]) -> None:
        resp = client.get("/api/ai/doctors/1/availability", headers=auth)
        assert resp.status_code == 200
        assert resp.json() == []


class TestMonitoringEndpoint:
    """The authenticated metrics summary returns the HokuMetrics dict."""

    def test_monitoring_metrics_summary(self, client: TestClient, auth: Dict[str, str]) -> None:
        resp = client.get("/api/ai/monitoring/metrics", headers=auth)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "requests_total" in body["metrics"]


class TestRagDebugEndpoint:
    """The RAG debug search endpoint responds without an LLM call."""

    def test_rag_search_returns_results_key(self, client: TestClient) -> None:
        resp = client.get("/api/ai/rag/search?q=services")
        # 200 with a results list (empty on an unseeded store) is the contract.
        assert resp.status_code == 200
        assert "results" in resp.json()


# ---------------------------------------------------------------------------
# SafetyLog CRUD (audit persistence)
# ---------------------------------------------------------------------------
class TestSafetyCrud:
    """Direct exercise of the SafetyLog audit-logging CRUD layer."""

    def test_log_and_fetch_by_user(self, db: Any) -> None:
        from app.crud.crud_safety import (
            get_safety_logs_by_user,
            log_safety_violation,
        )

        Base.metadata.create_all(bind=engine)
        log_safety_violation(
            db=db,
            user_id=USER_ID,
            message="test message",
            ai_response="[audit]",
            violation_type="diagnosis",
            severity="high",
        )
        logs = get_safety_logs_by_user(db, user_id=USER_ID)
        assert len(logs) >= 1
        assert logs[0].violation_type == "diagnosis"

    def test_fetch_by_type_and_severity(self, db: Any) -> None:
        from app.crud.crud_safety import (
            get_safety_logs_by_severity,
            get_safety_logs_by_type,
            log_safety_violation,
        )

        Base.metadata.create_all(bind=engine)
        log_safety_violation(
            db=db,
            user_id=USER_ID,
            message="another",
            ai_response="[audit]",
            violation_type="prescription",
            severity="moderate",
        )
        by_type = get_safety_logs_by_type(db, violation_type="prescription")
        by_sev = get_safety_logs_by_severity(db, severity="moderate")
        assert any(log.violation_type == "prescription" for log in by_type)
        assert any(log.severity == "moderate" for log in by_sev)
