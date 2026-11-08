"""symptom_extractor's LLM path (ChatGroq mocked) and the seed_faqs CLI."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.ai import symptom_extractor as se


@pytest.mark.parametrize(
    "reply,expected",
    [
        (
            AIMessage(content='{"symptoms": ["Dizziness", "nausea", "dizziness"]}'),
            ["dizziness", "nausea"],
        ),
        (AIMessage(content='```json\n{"symptoms": ["fatigue"]}\n```'), ["fatigue"]),
        ('{"symptoms": ["cramps"]}', ["cramps"]),
        ({"text": '{"symptoms": ["chills"]}'}, ["chills"]),
        (AIMessage(content="not json"), []),
        (AIMessage(content='{"symptoms": "not a list"}'), []),
    ],
)
def test_llm_extract_sync_parses_model_output(reply, expected):
    with patch("langchain_groq.ChatGroq") as chat:
        chat.return_value.invoke.return_value = reply
        assert se._llm_extract_symptoms_sync("I feel dizzy and sick") == expected


async def test_llm_extract_timeout_returns_empty(monkeypatch):
    async def _slow(*args, **kwargs):
        raise asyncio.TimeoutError

    monkeypatch.setattr(se.asyncio, "wait_for", _slow)
    assert await se._llm_extract_symptoms("text") == []


async def test_llm_extract_error_returns_empty(monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("x")

    monkeypatch.setattr(se.asyncio, "wait_for", _boom)
    assert await se._llm_extract_symptoms("text") == []


async def test_complex_text_uses_llm_result():
    with patch.object(se, "_llm_extract_symptoms", return_value=["dizziness"]) as llm:
        result = await se.extract_symptoms_from_text(
            "I feel dizzy and weak since morning"
        )
    assert result == ["dizziness"]
    llm.assert_awaited_once()


async def test_complex_text_falls_back_to_fever_when_llm_finds_nothing():
    with patch.object(se, "_llm_extract_symptoms", return_value=[]):
        assert await se.extract_symptoms_from_text("I feel unwell") == ["fever"]


async def test_llm_exception_falls_back_to_fever():
    with patch.object(se, "_llm_extract_symptoms", side_effect=RuntimeError("x")):
        assert await se.extract_symptoms_from_text("I feel unwell") == ["fever"]


def test_long_text_counts_as_complex():
    assert se._is_complex_text("word " * 60) is True
    assert se._is_complex_text("hello there") is False


def test_seed_faqs_main_reports_added_and_skipped(capsys):
    from app.scripts import seed_faqs

    rag = MagicMock()
    rag.collection_name = "hoku_health_faqs"
    rag.upsert_faq_documents.return_value = (3, 17)
    with patch.object(seed_faqs, "HokuRAG", return_value=rag):
        seed_faqs.main()
    rag.create_vector_store.assert_called_once()
    assert "Seeded 3 FAQ entries" in capsys.readouterr().out
    assert len(seed_faqs.FAQS) == 20
