"""Proxy-aware client IP and the per-IP limits (demo login, chat IP cap)."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from starlette.requests import Request

from app.api.v1.endpoints.auth import reset_demo_rate_limiter
from app.core.client_ip import client_ip
from app.core.config import Settings, settings
from app.core.rate_limit import reset_chat_rate_limiter


def _request(xff=None, peer="10.0.0.9"):
    headers = []
    for value in xff or []:
        headers.append((b"x-forwarded-for", value.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": (peer, 1234) if peer else None,
    }
    return Request(scope)


def _cfg(trust, hops=1):
    return SimpleNamespace(TRUST_PROXY=trust, TRUST_PROXY_HOPS=hops)


class TestClientIp:
    def test_untrusted_ignores_forwarded_header(self):
        req = _request(["1.2.3.4"])
        assert client_ip(req, _cfg(False)) == "10.0.0.9"

    def test_trusted_uses_right_most_entry(self):
        req = _request(["6.6.6.6, 1.2.3.4"])
        assert client_ip(req, _cfg(True)) == "1.2.3.4"

    def test_multiple_header_lines_are_joined(self):
        req = _request(["6.6.6.6", "1.2.3.4"])
        assert client_ip(req, _cfg(True)) == "1.2.3.4"

    def test_two_hops(self):
        req = _request(["6.6.6.6, 1.2.3.4, 35.1.1.1"])
        assert client_ip(req, _cfg(True, hops=2)) == "1.2.3.4"

    def test_too_few_entries_falls_back_to_peer(self):
        req = _request(["1.2.3.4"])
        assert client_ip(req, _cfg(True, hops=2)) == "10.0.0.9"

    def test_no_header_uses_peer(self):
        assert client_ip(_request(), _cfg(True)) == "10.0.0.9"

    def test_no_peer(self):
        assert client_ip(_request(peer=None), _cfg(False)) == "unknown"

    def test_hops_must_be_positive(self):
        with pytest.raises(ValidationError):
            Settings(TRUST_PROXY_HOPS=0)

    def test_defaults(self):
        fresh = Settings()
        assert fresh.TRUST_PROXY is False
        assert fresh.TRUST_PROXY_HOPS == 1
        assert fresh.RATE_LIMIT_REQUESTS_PER_MINUTE == 10
        assert fresh.CHAT_IP_RATE_PER_MINUTE == 20
        assert fresh.DEMO_RATE_LIMIT_PER_HOUR == 20


@pytest.fixture
def limits_on(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "TRUST_PROXY", True)
    monkeypatch.setattr(settings, "TRUST_PROXY_HOPS", 1)
    reset_chat_rate_limiter()
    reset_demo_rate_limiter()
    yield
    reset_chat_rate_limiter()
    reset_demo_rate_limiter()


def _chat(client, headers, xff):
    return client.post(
        "/api/ai/chat",
        json={"message": "hello", "userId": 1},
        headers={**headers, "X-Forwarded-For": xff},
    )


class TestChatIpCap:
    def test_cap_applies_across_spoofed_left_entries(
        self, client, auth_headers, limits_on, monkeypatch
    ):
        monkeypatch.setattr(settings, "CHAT_IP_RATE_PER_MINUTE", 2)
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 100)
        codes = [
            _chat(client, auth_headers, f"9.9.9.{i}, 1.2.3.4").status_code
            for i in range(3)
        ]
        assert codes == [200, 200, 429]

    def test_429_carries_headers(self, client, auth_headers, limits_on, monkeypatch):
        monkeypatch.setattr(settings, "CHAT_IP_RATE_PER_MINUTE", 1)
        _chat(client, auth_headers, "1.2.3.4")
        resp = _chat(client, auth_headers, "1.2.3.4")
        assert resp.status_code == 429
        assert resp.headers["Retry-After"]
        assert resp.headers["RateLimit-Remaining"] == "0"
        assert resp.headers["RateLimit-Limit"] == "1"

    def test_other_ip_has_its_own_budget(
        self, client, auth_headers, limits_on, monkeypatch
    ):
        monkeypatch.setattr(settings, "CHAT_IP_RATE_PER_MINUTE", 1)
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 100)
        assert _chat(client, auth_headers, "1.2.3.4").status_code == 200
        assert _chat(client, auth_headers, "5.6.7.8").status_code == 200
        assert _chat(client, auth_headers, "1.2.3.4").status_code == 429

    def test_per_user_limit_still_applies(
        self, client, auth_headers, limits_on, monkeypatch
    ):
        monkeypatch.setattr(settings, "CHAT_IP_RATE_PER_MINUTE", 100)
        monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS_PER_MINUTE", 1)
        assert _chat(client, auth_headers, "1.2.3.4").status_code == 200
        assert _chat(client, auth_headers, "5.6.7.8").status_code == 429

    def test_ip_logged_only_on_ip_429(
        self, client, auth_headers, limits_on, monkeypatch, caplog
    ):
        monkeypatch.setattr(settings, "CHAT_IP_RATE_PER_MINUTE", 1)
        caplog.set_level("INFO")
        _chat(client, auth_headers, "1.2.3.4")
        assert "1.2.3.4" not in caplog.text
        _chat(client, auth_headers, "1.2.3.4")
        assert "client IP 1.2.3.4" in caplog.text


class TestDemoLoginIp:
    def test_spoofed_left_entries_share_one_budget(
        self, client, limits_on, monkeypatch
    ):
        monkeypatch.setattr(settings, "DEMO_LOGIN_ENABLED", True)
        monkeypatch.setattr(settings, "DEMO_RATE_LIMIT_PER_HOUR", 2)
        codes = [
            client.post(
                "/api/auth/demo", headers={"X-Forwarded-For": f"9.9.9.{i}, 1.2.3.4"}
            ).status_code
            for i in range(3)
        ]
        assert codes == [200, 200, 429]

    def test_untrusted_proxy_ignores_header(self, client, limits_on, monkeypatch):
        monkeypatch.setattr(settings, "DEMO_LOGIN_ENABLED", True)
        monkeypatch.setattr(settings, "TRUST_PROXY", False)
        monkeypatch.setattr(settings, "DEMO_RATE_LIMIT_PER_HOUR", 1)
        first = client.post("/api/auth/demo", headers={"X-Forwarded-For": "1.1.1.1"})
        second = client.post("/api/auth/demo", headers={"X-Forwarded-For": "2.2.2.2"})
        assert (first.status_code, second.status_code) == (200, 429)
