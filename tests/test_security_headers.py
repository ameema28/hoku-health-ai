"""nosniff on every response, and a narrow, non-credentialed CORS default."""

from app.core.config import Settings


def test_nosniff_on_json_and_error_responses(client):
    for resp in (
        client.get("/health"),
        client.post("/api/ai/chat", json={"message": "hi", "userId": 1}),
    ):
        assert resp.headers["X-Content-Type-Options"] == "nosniff"


def test_default_cors_origins_are_local_only(monkeypatch):
    monkeypatch.delenv("CORS_ALLOW_ORIGINS", raising=False)
    origins = Settings().cors_origins
    assert origins == ["http://localhost:5173", "http://localhost:3000"]


def test_cors_preflight_is_not_credentialed(client):
    resp = client.options(
        "/api/ai/chat",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "access-control-allow-credentials" not in resp.headers


def test_unknown_origin_is_not_allowed(client):
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
