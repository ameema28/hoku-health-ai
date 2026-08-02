"""
Unit tests for the RateLimit header contract on POST /api/ai/chat.
"""



def test_rate_limit_headers_present_when_disabled(client, auth_headers, monkeypatch):
    """
    When RATE_LIMIT_ENABLED=false, headers must still be present
    so the API contract is stable.
    """
    monkeypatch.setattr("app.api.v1.endpoints.ai.settings.RATE_LIMIT_ENABLED", False)

    resp = client.post(
        "/api/ai/chat",
        json={"message": "hello", "userId": 1},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert resp.headers.get("RateLimit-Limit") is not None
    assert resp.headers.get("RateLimit-Remaining") is not None


def test_rate_limit_headers_present_when_enabled(client, auth_headers, monkeypatch):
    """
    When RATE_LIMIT_ENABLED=true, successful chat responses carry
    RateLimit-Limit and RateLimit-Remaining.
    Covers the enabled path that CI skips when RATE_LIMIT_ENABLED=false.
    """
    monkeypatch.setattr("app.api.v1.endpoints.ai.settings.RATE_LIMIT_ENABLED", True)

    from app.core.rate_limit import reset_chat_rate_limiter
    reset_chat_rate_limiter()

    resp = client.post(
        "/api/ai/chat",
        json={"message": "hello", "userId": 1},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert resp.headers.get("RateLimit-Limit") is not None
    assert resp.headers.get("RateLimit-Remaining") is not None
    assert int(resp.headers["RateLimit-Limit"]) > 0


def test_rate_limit_429_headers(client, auth_headers, monkeypatch):
    """
    Throttled requests carry RateLimit-Limit, RateLimit-Remaining=0,
    Retry-After, and RateLimit-Reset.
    """
    monkeypatch.setattr("app.api.v1.endpoints.ai.settings.RATE_LIMIT_ENABLED", True)

    from app.core.rate_limit import get_chat_rate_limiter, reset_chat_rate_limiter
    reset_chat_rate_limiter()

    limiter = get_chat_rate_limiter()
    original_limit = limiter.limit
    limiter.limit = 2

    try:
        # Exhaust the limit
        for _ in range(2):
            client.post(
                "/api/ai/chat",
                json={"message": "x", "userId": 1},
                headers=auth_headers,
            )

        resp = client.post(
            "/api/ai/chat",
            json={"message": "x", "userId": 1},
            headers=auth_headers,
        )

        assert resp.status_code == 429
        assert resp.headers.get("RateLimit-Limit") is not None
        assert resp.headers.get("RateLimit-Remaining") == "0"
        assert resp.headers.get("Retry-After") is not None
        assert resp.headers.get("RateLimit-Reset") is not None
    finally:
        limiter.limit = original_limit