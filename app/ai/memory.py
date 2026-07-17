"""
Hoku Health Care - Conversation Memory Management (Day 4).

Per-user conversation memory using LangChain 0.2.6 ConversationBufferMemory.
Loads recent chat history from PostgreSQL/SQLite and converts to LangChain
message format for context-aware multi-turn conversations.

Day 4 update:
- save_memory now accepts intent parameter for analytics
"""

import logging
import time
from typing import List, Optional

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy.orm import Session

from app.ai.config import ai_settings
from app.ai.token_budget import trim_history_to_budget
from app.crud.chat import create_chat_history, get_recent_chat_history

logger = logging.getLogger(__name__)


class HokuConversationMemory:
    """
    Manages per-user conversation memory for the Hoku AI chatbot.

    Why ConversationBufferMemory over SummaryMemory?
    - ConversationBufferMemory preserves exact wording of previous turns,
      which is critical for clinical accuracy (symptom descriptions must
      not be paraphrased or lost).
    - SummaryMemory compresses history, risking loss of nuanced symptom
      details that could affect safety assessments.
    - Trade-off: higher token usage. We mitigate this with a strict
      token budget (307 tokens) and message limit (10 turns).

    We limit to 10 messages because:
    - Token budget: 10 turns ≈ 20 messages (human + AI) ≈ 200–300 tokens,
      fitting comfortably inside the 307-token history budget.
    - Latency: Loading >10 turns from DB + formatting adds measurable
      overhead. With a 3.5s hard timeout, memory load must stay under
      ~200ms to leave time for the Groq LLM call.
    - Privacy: Shorter retention per request reduces exposure window.
    """

    def __init__(
        self,
        message_limit: int = 10,
        max_history_tokens: int = 307,
    ) -> None:
        """
        Initialize memory manager.

        Args:
            message_limit: Maximum number of recent turns to load from DB.
                10 turns balances context richness with token budget and
                keeps memory load time under the 3.5s total budget.
            max_history_tokens: Token budget for history. Leaves ~60% of
                the 512-token context window for the current response.
        """
        self.message_limit = message_limit
        self.max_history_tokens = max_history_tokens

    def load_memory(
        self,
        user_id: int,
        db: Session,
    ) -> ConversationBufferMemory:
        """
        Load conversation memory for a specific user from the database.

        Steps:
        1. Fetch last N complete turns from chat_history (ai_response NOT NULL).
        2. Reverse to chronological order (oldest first for memory buildup).
        3. Convert DB records to LangChain HumanMessage/AIMessage pairs.
        4. Trim to token budget if needed (oldest messages dropped first).
        5. Inject into ConversationBufferMemory with return_messages=True
           for MessagesPlaceholder compatibility.

        Args:
            user_id: The authenticated user's database ID.
            db: SQLAlchemy database session.

        Returns:
            ConversationBufferMemory: LangChain memory object ready for
                injection into LLMChain via MessagesPlaceholder.
        """
        memory_start = time.perf_counter()

        # Fetch complete turns only (ai_response is not NULL)
        db_entries = get_recent_chat_history(
            db=db,
            user_id=user_id,
            limit=self.message_limit,
        )

        # Reverse to chronological order (oldest first) for memory buildup
        db_entries = list(reversed(db_entries))

        # Convert to LangChain message format
        messages: List[BaseMessage] = []
        for entry in db_entries:
            if entry.message:
                messages.append(HumanMessage(content=entry.message))
            if entry.ai_response:
                messages.append(AIMessage(content=entry.ai_response))

        # Trim to token budget (drop oldest first if over budget)
        trimmed_messages = trim_history_to_budget(
            messages=messages,
            max_tokens=self.max_history_tokens,
        )

        # Build ConversationBufferMemory with return_messages=True.
        # This returns BaseMessage objects that MessagesPlaceholder
        # injects cleanly into ChatPromptTemplate (langchain 0.2.6).
        memory = ConversationBufferMemory(
            memory_key="history",
            input_key="message",
            return_messages=True,
        )

        # Manually populate the chat memory from DB records
        for msg in trimmed_messages:
            if isinstance(msg, HumanMessage):
                memory.chat_memory.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                memory.chat_memory.add_ai_message(msg.content)

        memory_elapsed = time.perf_counter() - memory_start
        token_count = self._estimate_token_count(trimmed_messages)

        logger.info(
            "Memory loaded for user_id=%s: %d turns, ~%d tokens, %.3fs",
            user_id,
            len(db_entries),
            token_count,
            memory_elapsed,
        )

        return memory

    def save_memory(
        self,
        user_id: int,
        human_message: str,
        ai_message: str,
        db: Session,
        intent: Optional[str] = None,  # Day 4: Accept intent for analytics
    ) -> None:
        """
        Persist a conversation turn to the database.

        This is a coordination wrapper around the CRUD layer to ensure
        memory state and database state stay synchronized.

        Args:
            user_id: The authenticated user's database ID.
            human_message: The user's message.
            ai_message: The AI's response.
            db: SQLAlchemy database session.
            intent: Classified intent string (Day 4). Defaults to "general"
                if not provided.
        """
        # Day 4: Use provided intent or default to "general"
        intent_to_save = intent if intent is not None else "general"

        create_chat_history(
            db=db,
            user_id=user_id,
            message=human_message,
            ai_response=ai_message,
            intent=intent_to_save,
        )

        logger.debug(
            "Memory saved for user_id=%s: message_len=%d, response_len=%d, intent=%s",
            user_id,
            len(human_message),
            len(ai_message),
            intent_to_save,
        )

    @staticmethod
    def _estimate_token_count(messages: List[BaseMessage]) -> int:
        """
        Estimate token count for a list of messages.

        Uses tiktoken if available, otherwise falls back to len/4 heuristic.
        The len/4 approximation is conservative for English text and
        safe for our 307-token budget on Windows where tiktoken may fail.
        """
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base")
            total = 0
            for msg in messages:
                total += len(encoding.encode(msg.content))
                # Each message has ~4 tokens of overhead (role tags, separators)
                total += 4
            return total
        except ImportError:
            # Windows fallback: character count / 4
            # For English text, 1 token ≈ 4 characters on average.
            total = 0
            for msg in messages:
                total += len(msg.content) // 4
                total += 4  # role metadata estimate
            return total