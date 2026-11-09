"""
Every Groq client is built with an explicit timeout and max_retries=0.

asyncio.wait_for cannot stop the worker thread of a call the request budget
abandoned. With no retries and a client timeout, that thread ends within the
stage's own timeout instead of retrying in the background. No network: the
clients are only constructed, never invoked against Groq.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.ai.config import ai_settings
from app.core.config import settings as core_settings


@pytest.fixture
def fake_key(monkeypatch):
    monkeypatch.setattr(core_settings, "GROQ_API_KEY", "gsk_test_not_a_real_key")


def _kwargs(mock_cls: MagicMock) -> dict:
    assert mock_cls.call_count == 1
    return mock_cls.call_args.kwargs


def test_groq_sdk_and_chatgroq_construct_offline():
    """
    The installed groq SDK and httpx must be compatible. groq 0.9.0 passes
    proxies= to httpx.Client, which httpx 0.28 removed; with that pair no
    client can be created and every reply silently becomes a fallback.
    """
    import groq
    from langchain_core.utils import convert_to_secret_str
    from langchain_groq import ChatGroq

    groq.Groq(api_key="gsk_test_not_a_real_key")
    ChatGroq(
        model=ai_settings.GROQ_MAIN_MODEL,
        api_key=convert_to_secret_str("gsk_test_not_a_real_key"),
        timeout=1.0,
        max_retries=0,
    )


def test_fast_model_setting_drives_intent_and_symptom_models(monkeypatch):
    """One setting to change if Groq retires the fast model."""
    from app.ai.config import AISettings

    for name in ("INTENT_MODEL", "SYMPTOM_EXTRACTION_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GROQ_FAST_MODEL", "new-fast-model")
    cfg = AISettings()
    assert cfg.INTENT_MODEL == "new-fast-model"
    assert cfg.SYMPTOM_EXTRACTION_MODEL == "new-fast-model"

    monkeypatch.setenv("INTENT_MODEL", "intent-only")
    assert AISettings().INTENT_MODEL == "intent-only"
    assert AISettings().SYMPTOM_EXTRACTION_MODEL == "new-fast-model"


def test_factory_fast_llm_real_client_has_no_retries(fake_key):
    from app.ai.llm_optimizer import LLMFactory

    llm = LLMFactory.get_fast_llm(api_key="gsk_test", request_timeout=0.8)
    assert llm is not None
    assert llm.max_retries == 0
    assert llm.request_timeout == 0.8


def test_factory_main_llm_real_client_has_no_retries(fake_key):
    from app.ai.llm_optimizer import LLMFactory

    llm = LLMFactory.get_main_llm(api_key="gsk_test")
    assert llm is not None
    assert llm.max_retries == 0
    assert llm.request_timeout == ai_settings.GROQ_TIMEOUT_SECONDS


def test_intent_classifier_client(fake_key):
    from app.ai.intent_classifier import IntentClassifier

    with patch("app.ai.intent_classifier.ChatGroq") as mock_cls:
        classifier = IntentClassifier()
        classifier.groq_api_key = "gsk_test"
        assert classifier.llm is mock_cls.return_value
    kwargs = _kwargs(mock_cls)
    assert kwargs["max_retries"] == 0
    assert kwargs["timeout"] == ai_settings.INTENT_CLASSIFICATION_TIMEOUT


def test_symptom_extractor_client(fake_key):
    from app.ai.symptom_extractor import _llm_extract_symptoms_sync

    with patch("langchain_groq.ChatGroq") as mock_cls:
        mock_cls.return_value.invoke.return_value = '{"symptoms": ["cough"]}'
        assert _llm_extract_symptoms_sync("I have a cough") == ["cough"]
    kwargs = _kwargs(mock_cls)
    assert kwargs["max_retries"] == 0
    assert kwargs["timeout"] == ai_settings.SYMPTOM_EXTRACTION_TIMEOUT


def test_deploy_check_client(fake_key):
    from app.scripts.deploy_check import check_groq

    with patch("langchain_groq.ChatGroq") as mock_cls:
        check_groq(live=False)
    kwargs = _kwargs(mock_cls)
    assert kwargs["max_retries"] == 0
    assert kwargs["timeout"] == ai_settings.GROQ_TIMEOUT_SECONDS
