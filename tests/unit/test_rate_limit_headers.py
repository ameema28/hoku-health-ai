def test_rate_limit_headers_present_when_disabled(client, auth_headers, monkeypatch):
    monkeypatch.setattr("app.api.v1.endpoints.ai.settings.RATE_LIMIT_ENABLED", False)
    resp = client.post(
        "/api/ai/chat",
        json={"message": "hello", "userId": 1},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.headers.get("RateLimit-Limit") is not None
    assert resp.headers.get("RateLimit-Remaining") is not None