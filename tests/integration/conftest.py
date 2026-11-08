# tests/integration/conftest.py
"""Fixtures scoped to integration tests only — cannot leak to unit tests."""

from unittest.mock import patch
import pytest


@pytest.fixture(autouse=True)
def mock_ai_boundaries():
    """Force Groq/RAG boundaries offline for integration tests only."""
    from app.ai.intent_classifier import IntentEnum

    async def _fake_classify_intent(self, message: str):
        lowered = (message or "").lower()
        if (
            "chest pain" in lowered
            or "can't breathe" in lowered
            or "cant breathe" in lowered
        ):
            return (IntentEnum.EMERGENCY, 0.99)
        return (IntentEnum.GENERAL, 0.95)

    from langchain_core.documents import Document

    def _fake_retrieve(self, query, threshold=None):
        return [
            Document(
                page_content="Q: What services does Hoku offer?",
                metadata={
                    "question": "What services does Hoku offer?",
                    "answer": "Home health, palliative, and hospice care.",
                    "score": 0.9,
                },
            )
        ]

    with patch(
        "app.ai.intent_classifier.IntentClassifier.classify_intent",
        new=_fake_classify_intent,
    ), patch(
        "app.ai.rag.HokuRAG.retrieve",
        new=_fake_retrieve,
    ):
        yield
