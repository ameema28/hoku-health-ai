"""No patient text, tokens or keys in log records (checked before any redaction filter)."""

import io
import json
import logging
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from app.ai.chatbot import HokuChatbot
from app.ai.intent_classifier import IntentClassifier, IntentEnum
from app.core.config import settings
from app.core.logging import RedactionFilter
from app.utils.constants import SAFETY_DISCLAIMER
from tests.auth_utils import add_user, bearer

MARKER = "ZQXMARKER7731"
DUMMY_GROQ_KEY = "gsk_" + "D" * 40  # dummy, never sent anywhere (no network in tests)


def _all_text(caplog) -> str:
    formatter = logging.Formatter("%(name)s %(levelname)s %(message)s")
    return "\n".join(formatter.format(r) + " " + repr(r.args) for r in caplog.records)


def _chain(reply=None, error=None):
    chain = MagicMock()
    if error is not None:
        chain.invoke.side_effect = error
    else:
        chain.invoke.return_value = json.dumps(
            reply
            or {
                "reply": "Please rest. " + SAFETY_DISCLAIMER,
                "suggestedSpecialist": "General Physician",
                "severity": "mild",
                "shouldSeeDoctor": False,
            }
        )
    return chain


@pytest.fixture
def chat(client, db, monkeypatch, caplog):
    monkeypatch.setattr(settings, "GROQ_API_KEY", DUMMY_GROQ_KEY)
    add_user(db, user_id=1, role="patient")
    token = bearer(1, "patient")
    caplog.set_level(logging.DEBUG)

    def _send(message, chain, intent=IntentEnum.SYMPTOM):
        with (
            # A real (offline) main LLM object is not needed: the chain is faked.
            patch.object(
                HokuChatbot,
                "main_llm",
                new_callable=PropertyMock,
                return_value=MagicMock(),
            ),
            patch.object(HokuChatbot, "_generation_chain", return_value=chain),
            patch.object(
                IntentClassifier, "classify_intent", return_value=(intent, 0.95)
            ),
            patch("app.ai.chatbot.HokuRAG"),
        ):
            resp = client.post(
                "/api/ai/chat",
                json={"message": message, "userId": 1},
                headers={"Authorization": token},
            )
        return resp, token

    return _send


def _assert_clean(caplog, token):
    text = _all_text(caplog)
    assert caplog.records, "expected some log records"
    assert MARKER not in text
    assert token.split()[1] not in text
    assert DUMMY_GROQ_KEY not in text


def test_symptom_path_logs_no_message_or_symptoms(chat, caplog):
    resp, token = chat(
        f"{MARKER} I have had a fever and a headache since yesterday", _chain()
    )
    assert resp.status_code == 200
    _assert_clean(caplog, token)
    assert "fever" not in _all_text(caplog).lower()


def test_emergency_path_logs_no_message(chat, caplog):
    resp, token = chat(f"{MARKER} I can't breathe and my chest hurts", _chain())
    assert resp.status_code == 200
    assert resp.headers.get("X-Hoku-Emergency")
    _assert_clean(caplog, token)
    assert "breathe" not in _all_text(caplog).lower()


def test_exception_path_logs_no_exception_text(chat, caplog):
    error = RuntimeError(f"provider echoed the prompt: {MARKER}")
    chain = _chain(error=error)
    resp, token = chat(f"{MARKER} general wellness question", chain)
    assert resp.status_code == 200
    assert chain.invoke.called, "the failing chain was never reached"
    _assert_clean(caplog, token)
    assert "RuntimeError" in _all_text(caplog)  # the type is still logged


def test_redaction_filter_scrubs_emitted_output():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactionFilter())
    log = logging.getLogger("tests.redaction")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    jwt_like = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJlLXZhbHVl"
    secrets = [
        jwt_like,
        DUMMY_GROQ_KEY,
        "patient.one@example.com",
        "+92 300 1234567",
        "db-pass-123",
    ]
    try:
        log.info(
            "token=%s key=%s email=%s phone=%s url=%s",
            jwt_like,
            DUMMY_GROQ_KEY,
            "patient.one@example.com",
            "+92 300 1234567",
            "postgresql://hoku:db-pass-123@db:5432/hoku",
        )
    finally:
        log.removeHandler(handler)
    output = stream.getvalue()
    assert output.strip()
    for value in secrets:
        assert value not in output


@pytest.mark.parametrize(
    "env,debug,expected",
    [
        ("production", True, False),
        ("development", True, True),
        ("production", False, False),
    ],
)
def test_sql_echo_is_never_on_in_production(env, debug, expected):
    from app.core.config import Settings
    from app.core.database import sql_echo_enabled

    cfg = Settings(_env_file=None, ENVIRONMENT=env, DEBUG=debug)
    assert sql_echo_enabled(cfg) is expected
