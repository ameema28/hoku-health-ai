"""The demo page at GET /: CSP, content, examples and the demo-login flow."""

import base64
import json
import re

import pytest

from app.ai.emergency_detector import EmergencyDetector
from app.ai.rag import HokuRAG
from app.ai.specialist_mapper import SpecialistMapper
from app.ai.symptom_extractor import _regex_extract_symptoms
from app.api.v1.endpoints.auth import reset_demo_rate_limiter
from app.core.config import settings
from app.demo_page import DEMO_PAGE_HTML, csp_hash, load_demo_page
from app.scripts.seed_faqs import FAQS

EXAMPLES = re.findall(
    r'<button type="button">([^<]+)</button>',
    DEMO_PAGE_HTML.split('id="examples"', 1)[1].split("</div>", 1)[0],
)
EMERGENCY, SYMPTOM, FAQ, BOOKING = EXAMPLES


def _script() -> str:
    return re.search(r"<script>(.*?)</script>", DEMO_PAGE_HTML, re.DOTALL).group(1)


class TestPageAndHeaders:
    def test_served_as_html_with_hash_csp(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        csp = resp.headers["content-security-policy"]
        script = _script()
        style = re.search(r"<style>(.*?)</style>", DEMO_PAGE_HTML, re.DOTALL).group(1)
        assert f"script-src {csp_hash(script)}" in csp
        assert f"style-src {csp_hash(style)}" in csp
        assert "default-src 'none'" in csp
        assert "connect-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "unsafe-inline" not in csp
        assert resp.headers["x-content-type-options"] == "nosniff"
        assert resp.headers["referrer-policy"] == "no-referrer"

    def test_not_in_openapi_schema(self, client):
        assert "/" not in client.get("/openapi.json").json()["paths"]

    def test_extra_inline_script_is_refused(self, tmp_path):
        page = tmp_path / "index.html"
        page.write_text(
            "<style>a{}</style><script>1</script><script>2</script>", encoding="utf-8"
        )
        with pytest.raises(RuntimeError):
            load_demo_page(page)

    def test_banner_says_not_medical_advice(self):
        assert "Not medical advice." in DEMO_PAGE_HTML
        assert 'id="banner"' in DEMO_PAGE_HTML

    def test_calls_demo_login_and_chat_on_same_origin(self):
        script = _script()
        assert '"/api/auth/demo"' in script
        assert '"/api/ai/chat"' in script
        assert "http" not in script.split("LOGIN_URL", 1)[1].split(";", 1)[0]

    def test_token_kept_in_memory_only(self):
        script = _script()
        for api in ("localStorage", "sessionStorage", "indexedDB", "document.cookie"):
            assert api not in script

    def test_server_text_inserted_as_text_only(self):
        assert "innerHTML" not in _script()
        assert "insertAdjacentHTML" not in _script()

    def test_details_show_only_api_fields(self):
        script = _script()
        used = set(re.findall(r"data\.([a-zA-Z_]+)", script))
        assert used <= {
            "reply",
            "intent",
            "confidence",
            "severity",
            "shouldSeeDoctor",
            "suggestedSpecialist",
            "doctor_suggestion",
            "fallback_used",
        }
        headers = set(re.findall(r'headers\.get\("([^"]+)"\)', script))
        assert headers == {"X-Hoku-Emergency", "X-Response-Time-Sec"}

    def test_mobile_rules_present(self):
        assert "max-width: 600px" in DEMO_PAGE_HTML
        assert "overflow-x: auto" in DEMO_PAGE_HTML
        assert "showNewest" in _script()


class TestExamples:
    def test_four_examples(self):
        assert len(EXAMPLES) == 4

    def test_emergency_example_hits_tier1(self):
        is_emergency, urgency, _ = EmergencyDetector.detect_emergency(EMERGENCY)
        assert is_emergency is True
        assert urgency == "high"

    @pytest.mark.parametrize("message", [SYMPTOM, FAQ, BOOKING])
    def test_other_examples_are_not_emergencies(self, message):
        assert EmergencyDetector.detect_emergency(message)[0] is False

    def test_symptom_example_maps_to_seeded_specialty(self):
        symptoms = _regex_extract_symptoms(SYMPTOM)
        assert SpecialistMapper.map_symptoms_to_specialist(symptoms) == "Dermatologist"

    @pytest.mark.parametrize(
        "message,expected_question",
        [
            (FAQ, "Do you provide home nursing care in Lahore?"),
            (BOOKING, "What is the typical wait time for a home visit booking?"),
        ],
    )
    def test_faq_examples_retrieve_expected_faq(self, db, message, expected_question):
        rag = HokuRAG(db=db)
        rag.upsert_faq_documents(FAQS)
        docs = rag.retrieve(message)
        assert docs, "no FAQ above the threshold"
        assert docs[0].metadata["question"] == expected_question


def _sub(token: str) -> int:
    part = token.split(".")[1]
    part += "=" * (-len(part) % 4)
    return int(json.loads(base64.urlsafe_b64decode(part))["sub"])


def test_page_flow_demo_login_then_chat(client, monkeypatch):
    """What the page does: demo login, user id from the token's sub, chat."""
    monkeypatch.setattr(settings, "DEMO_LOGIN_ENABLED", True)
    reset_demo_rate_limiter()
    token = client.post("/api/auth/demo").json()["access_token"]
    resp = client.post(
        "/api/ai/chat",
        json={"message": EMERGENCY, "userId": _sub(token)},
        headers={"Authorization": f"Bearer {token}"},
    )
    reset_demo_rate_limiter()
    assert resp.status_code == 200
    assert resp.headers["X-Hoku-Emergency"] == "true"
