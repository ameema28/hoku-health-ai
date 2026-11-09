"""smoke_check.py and benchmark.py against a fake HTTP layer (no network)."""

import base64
import json

import pytest

from scripts import benchmark as bm
from scripts import smoke_check as sc

DISCLAIMER = "Please consult a doctor for proper diagnosis."


def _token(sub=7):
    payload = base64.urlsafe_b64encode(json.dumps({"sub": str(sub)}).encode()).decode()
    return f"h.{payload.rstrip('=')}.s"


class FakeService:
    """Mimics the deployed API closely enough for the checks."""

    def __init__(self, fallback=False, decline=True):
        self.fallback = fallback
        self.decline = decline
        self.user_hits = {}
        self.ip_hits = 0

    def new_window(self):
        """What the 61 s wait achieves on the real service."""
        self.user_hits = {}
        self.ip_hits = 0

    def __call__(self, method, url, body=None, headers=None, timeout=90):
        headers = headers or {}
        path = (
            url.split("://", 1)[1].split("/", 1)[1]
            if "/" in url.split("://", 1)[1]
            else ""
        )
        path = "/" + path
        if path == "/health":
            return 200, {}, json.dumps({"status": "ok", "rag_ready": True}), 0.01
        if path == "/":
            html = f"{sc.PAGE_MARKER} Not medical advice."
            return (
                200,
                {"content-security-policy": "script-src 'sha256-abc'"},
                html,
                0.01,
            )
        if path == sc.LOGIN_PATH:
            self.logins = getattr(self, "logins", 0) + 1
            return 200, {}, json.dumps({"access_token": _token(self.logins)}), 0.01
        if path == sc.CHAT_PATH:
            if "Authorization" not in headers:
                return 401, {}, "{}", 0.01
            if body["message"] == "":
                self.ip_hits += 1
                key = headers["Authorization"]
                self.user_hits[key] = self.user_hits.get(key, 0) + 1
                if (
                    self.ip_hits > sc.CHAT_IP_LIMIT
                    or self.user_hits[key] > sc.CHAT_LIMIT
                ):
                    return 429, {}, "{}", 0.01
                return 422, {}, "{}", 0.01
            return self._chat(body["message"])
        return 404, {}, "{}", 0.01

    def _chat(self, message):
        resp_headers = {"x-response-time-sec": "1.250"}
        data = {"fallback_used": self.fallback, "doctor_suggestion": None}
        if message == sc.EMERGENCY_EXAMPLE:
            resp_headers["x-hoku-emergency"] = "true"
            data.update(
                intent="emergency",
                fallback_used=False,
                reply=f"Call 1122. {DISCLAIMER}",
            )
        elif "rash" in message:
            data.update(
                intent="symptom",
                reply=f"A dermatologist can help. {DISCLAIMER}",
                doctor_suggestion={"specialist": "Dermatologist"},
            )
        elif "booking" in message:
            data.update(
                intent="booking",
                reply=f"Visits are confirmed in 2-4 hours. {DISCLAIMER}",
            )
        elif any(word in message for word in ("mall", "Amazon")):
            text = (
                "I can only help with health questions."
                if self.decline
                else "Usually 2-4 hours."
            )
            data.update(intent="general", reply=f"{text} {DISCLAIMER}")
        else:
            data.update(intent="general", reply=f"Yes, in Lahore. {DISCLAIMER}")
        return 200, resp_headers, json.dumps(data), 0.5


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(sc.time, "sleep", lambda s: None)


def _run(monkeypatch, service, **kwargs):
    monkeypatch.setattr(sc, "http_request", service)
    monkeypatch.setattr(sc, "_wait", service.new_window)
    return {r["check"]: r["status"] for r in sc.run("http://svc", **kwargs)}


class TestSmokeCheck:
    def test_all_pass_against_healthy_service(self, monkeypatch, no_sleep):
        statuses = _run(monkeypatch, FakeService())
        assert set(statuses.values()) == {"PASS"}, statuses
        assert "per-user chat limit" in statuses and "per-IP chat cap" in statuses

    def test_fallback_replies_fail_the_examples(self, monkeypatch, no_sleep):
        statuses = _run(monkeypatch, FakeService(fallback=True), skip_rate_limit=True)
        assert statuses["example: FAQ"] == "FAIL"
        assert statuses["example: emergency"] == "PASS"  # never a fallback
        assert "per-user chat limit" not in statuses

    def test_scope_probe_is_warn_only(self, monkeypatch, no_sleep):
        statuses = _run(monkeypatch, FakeService(decline=False), skip_rate_limit=True)
        assert statuses["scope: shopping mall"] == "WARN"
        assert "FAIL" not in statuses.values()

    def test_demo_limit_check_is_opt_in(self, monkeypatch, no_sleep):
        statuses = _run(monkeypatch, FakeService(), skip_rate_limit=True)
        assert "demo login limit" not in statuses

    def test_examples_match_the_demo_page(self):
        from app.demo_page import DEMO_PAGE_HTML

        for _, message, _, _ in sc.EXAMPLES:
            assert message in DEMO_PAGE_HTML

    def test_helpers(self):
        assert sc.token_subject(_token(42)) == 42
        assert sc.limit_reached([422, 422, 429], (422,))
        assert not sc.limit_reached([422, 429, 422], (422,))
        assert not sc.limit_reached([], (422,))
        assert sc.normalize_base_url("https://x.run.app/") == "https://x.run.app"
        with pytest.raises(ValueError):
            sc.normalize_base_url("x.run.app")

    def test_out_path_guard(self, tmp_path):
        with pytest.raises(ValueError):
            sc.resolve_out_path(str(sc.REPO_ROOT / "smoke.json"))
        assert sc.resolve_out_path(str(tmp_path / "s.json")) == tmp_path / "s.json"


class TestBenchmark:
    def test_percentile_nearest_rank(self):
        values = [5, 1, 4, 2, 3]
        assert bm.percentile(values, 50) == 3
        assert bm.percentile(values, 95) == 5
        assert bm.percentile([], 50) is None

    def test_interval_guard(self):
        assert bm.validate_interval(6.5, False) == 6.5
        with pytest.raises(ValueError):
            bm.validate_interval(2.0, False)
        assert bm.validate_interval(2.0, True) == 2.0

    def test_out_path_guard_and_default(self):
        with pytest.raises(ValueError):
            bm.resolve_out_path(str(bm.REPO_ROOT / "b.json"))
        assert bm.REPO_ROOT not in bm.resolve_out_path(None).parents

    def test_messages_exclude_emergency(self):
        assert sc.EMERGENCY_EXAMPLE not in bm.MESSAGES
        assert len(bm.MESSAGES) == 3

    def test_summary_counts_and_server_times(self):
        records = [
            {"status": 200, "client_s": 1.0, "server_s": 0.8, "fallback_used": False},
            {"status": 200, "client_s": 2.0, "server_s": 1.5, "fallback_used": True},
            {"status": 429, "client_s": 0.1, "server_s": None, "fallback_used": None},
        ]
        summary = bm.summarize(records)
        assert summary["requests"] == 3 and summary["ok"] == 2
        assert summary["fallback_used"] == 1
        assert summary["server_s"]["max"] == 1.5
