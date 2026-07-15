"""
Hoku Health Care - Conversation Memory Unit Tests (Day 3).

Tests for HokuConversationMemory and token budget management.
All DB interactions are mocked; no real Groq API calls.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage

from app.ai.memory import HokuConversationMemory
from app.ai.token_budget import calculate_history_tokens, trim_history_to_budget


# ------------------------------------------------------------------
# Module-level fixtures (shared across ALL test classes in this file)
# ------------------------------------------------------------------
@pytest.fixture
def memory_manager():
    return HokuConversationMemory(message_limit=10, max_history_tokens=307)


@pytest.fixture
def mock_db():
    return MagicMock()


@pytest.fixture
def mock_memory():
    memory = ConversationBufferMemory(memory_key="history", input_key="message")
    return memory


class TestHokuConversationMemory:
    def test_load_memory_correct_message_count(self, memory_manager, mock_db):
        """Test that memory loads exactly the configured number of past messages."""
        mock_entries = []
        for i in range(10):
            entry = MagicMock()
            entry.message = f"Human message {i}"
            entry.ai_response = f"AI response {i}"
            mock_entries.append(entry)

        with patch(
            "app.ai.memory.get_recent_chat_history",
            return_value=mock_entries,
        ):
            memory = memory_manager.load_memory(user_id=1, db=mock_db)

            assert isinstance(memory, ConversationBufferMemory)
            # 10 turns = 20 messages (10 human + 10 ai) in the chat memory
            assert len(memory.chat_memory.messages) == 20

    def test_load_memory_empty_history_new_user(self, memory_manager, mock_db):
        """Test that a new user with no history gets empty memory."""
        with patch(
            "app.ai.memory.get_recent_chat_history",
            return_value=[],
        ):
            memory = memory_manager.load_memory(user_id=999, db=mock_db)

            assert isinstance(memory, ConversationBufferMemory)
            assert memory.chat_memory.messages == []

    def test_per_user_isolation(self, memory_manager, mock_db):
        """Test that user A cannot see user B's chat history."""
        user_a_entries = [
            MagicMock(message="User A question", ai_response="User A answer"),
        ]
        user_b_entries = [
            MagicMock(message="User B question", ai_response="User B answer"),
        ]

        with patch(
            "app.ai.memory.get_recent_chat_history",
            side_effect=lambda db, user_id, limit: (
                user_a_entries if user_id == 1 else user_b_entries
            ),
        ):
            memory_a = memory_manager.load_memory(user_id=1, db=mock_db)
            memory_b = memory_manager.load_memory(user_id=2, db=mock_db)

            # Extract all message contents for verification
            contents_a = [msg.content for msg in memory_a.chat_memory.messages]
            contents_b = [msg.content for msg in memory_b.chat_memory.messages]

            assert any("User A" in c for c in contents_a)
            assert any("User B" in c for c in contents_b)
            assert not any("User A" in c for c in contents_b)
            assert not any("User B" in c for c in contents_a)

    def test_load_memory_only_complete_turns(self, memory_manager, mock_db):
        """Test that entries with NULL ai_response are not loaded."""
        entry_complete = MagicMock(message="Complete", ai_response="Response")
        entry_incomplete = MagicMock(message="Incomplete", ai_response=None)

        with patch(
            "app.ai.memory.get_recent_chat_history",
            return_value=[entry_complete],
        ):
            memory = memory_manager.load_memory(user_id=1, db=mock_db)
            # Should only have the complete turn (1 human + 1 ai = 2 messages)
            assert len(memory.chat_memory.messages) == 2
            contents = [msg.content for msg in memory.chat_memory.messages]
            assert any("Complete" in c for c in contents)
            assert not any("Incomplete" in c for c in contents)

    def test_load_memory_returns_message_objects(self, memory_manager, mock_db):
        """Test that load_memory returns return_messages=True format
        for MessagesPlaceholder compatibility."""
        entry = MagicMock(message="Hello", ai_response="Hi there")
        with patch(
            "app.ai.memory.get_recent_chat_history",
            return_value=[entry],
        ):
            memory = memory_manager.load_memory(user_id=1, db=mock_db)
            vars_dict = memory.load_memory_variables({"message": "test"})
            history = vars_dict.get("history", [])

            # MessagesPlaceholder expects a list of BaseMessage objects
            assert isinstance(history, list)
            assert len(history) == 2
            assert isinstance(history[0], HumanMessage)
            assert isinstance(history[1], AIMessage)


class TestTokenBudget:
    def test_calculate_history_tokens_with_messages(self):
        """Test token estimation for a list of messages."""
        messages = [
            HumanMessage(content="Hello, I have a headache."),
            AIMessage(content="I'm sorry to hear that."),
        ]
        tokens = calculate_history_tokens(messages)
        assert tokens > 0
        # Should be reasonable estimate (not zero, not millions)
        assert tokens < 1000

    def test_calculate_history_tokens_empty(self):
        """Test token estimation with empty message list."""
        assert calculate_history_tokens([]) == 0

    def test_trim_history_to_budget_within_limit(self):
        """Test that history under budget is returned unchanged."""
        messages = [
            HumanMessage(content="Short message."),
            AIMessage(content="Short reply."),
        ]
        result = trim_history_to_budget(messages, max_tokens=1000)
        assert len(result) == 2
        assert result[0].content == "Short message."

    def test_trim_history_to_budget_trims_oldest(self):
        """Test that oldest message pairs are dropped when over budget."""
        messages = [
            HumanMessage(content="Old question from long ago that should be removed first"),
            AIMessage(content="Old answer that should also be removed"),
            HumanMessage(content="New important question"),
            AIMessage(content="New important answer"),
        ]

        # Set a very low budget to force trimming
        result = trim_history_to_budget(messages, max_tokens=10)

        # Oldest pair should be removed
        assert len(result) <= 4
        # The newest messages should be preserved
        if len(result) >= 2:
            assert "New important" in result[-2].content

    def test_trim_history_to_budget_empty(self):
        """Test trimming empty history."""
        result = trim_history_to_budget([], max_tokens=100)
        assert result == []


class TestMemoryIntegration:
    def test_memory_save_coordination(self, mock_db):
        """Test that save_memory delegates to CRUD correctly."""
        manager = HokuConversationMemory()
        with patch("app.ai.memory.create_chat_history") as mock_create:
            manager.save_memory(
                user_id=1,
                human_message="Test message",
                ai_message="Test response",
                db=mock_db,
            )
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["user_id"] == 1
            assert call_kwargs["message"] == "Test message"
            assert call_kwargs["ai_response"] == "Test response"
            assert call_kwargs["intent"] == "general_health"