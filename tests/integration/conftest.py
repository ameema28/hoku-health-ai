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
        if "chest pain" in lowered or "can't breathe" in lowered or "cant breathe" in lowered:
            return (IntentEnum.EMERGENCY, 0.99)
        return (IntentEnum.GENERAL, 0.95)

    def _fake_build_context(self, query, threshold=None):
        return "- Q: What services does Hoku offer?\n  A: Home health, palliative, and hospice care."

    with patch(
        "app.ai.intent_classifier.IntentClassifier.classify_intent",
        new=_fake_classify_intent,
    ), patch(
        "app.ai.rag.HokuRAG.build_context",
        new=_fake_build_context,
    ):
        yield