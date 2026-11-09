"""Token counting falls back to len/4 when tiktoken cannot load its encoding
(for example offline, without the cached cl100k_base file)."""

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from app.ai.memory import HokuConversationMemory
from app.ai.token_budget import calculate_history_tokens

MESSAGES = [HumanMessage(content="x" * 40), AIMessage(content="y" * 80)]
EXPECTED = (40 // 4 + 4) + (80 // 4 + 4)


def _offline(*args, **kwargs):
    raise ConnectionError("no network")


def test_token_budget_falls_back_when_encoding_unavailable():
    with patch("tiktoken.get_encoding", side_effect=_offline):
        assert calculate_history_tokens(MESSAGES) == EXPECTED


def test_memory_estimate_falls_back_when_encoding_unavailable():
    with patch("tiktoken.get_encoding", side_effect=_offline):
        assert HokuConversationMemory._estimate_token_count(MESSAGES) == EXPECTED
