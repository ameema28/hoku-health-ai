"""
Hoku Health Care - Token Budget Management (Day 3).

Token counting and history trimming to ensure we stay within the Groq
context window while preserving the most recent (and thus most clinically
relevant) conversation turns.
"""

import logging
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)


def calculate_history_tokens(messages: List[BaseMessage]) -> int:
    """
    Estimate the total token count for a list of chat messages.

    Uses tiktoken (cl100k_base) when available for accurate counting.
    Falls back to len(text)/4 heuristic on Windows where tiktoken may
    fail to install due to Rust/C extension compilation issues.

    Args:
        messages: List of LangChain BaseMessage objects (HumanMessage, AIMessage).

    Returns:
        int: Estimated total token count including role metadata overhead.
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
        # This is conservative and safe for budget management.
        total = 0
        for msg in messages:
            total += len(msg.content) // 4
            total += 4  # role metadata estimate
        return total


def trim_history_to_budget(
    messages: List[BaseMessage],
    max_tokens: int,
) -> List[BaseMessage]:
    """
    Trim conversation history to fit within a token budget.

    Strategy: Drop the OLDEST message pairs first while preserving the
    most recent context. This is clinically sound because recent symptoms
    and clarifications are typically more relevant than earlier turns.

    We trim in pairs (Human + AI) to maintain conversation coherence.
    If a single message exceeds the budget, we keep it but log a warning
    (dropping it would leave the user with zero context).

    Args:
        messages: Chronologically ordered list of messages (oldest first).
        max_tokens: Maximum allowed tokens for the history portion.

    Returns:
        List[BaseMessage]: Trimmed message list within budget.
    """
    if not messages:
        return []

    # If already under budget, return as-is
    current_tokens = calculate_history_tokens(messages)
    if current_tokens <= max_tokens:
        logger.debug(
            "History within budget: %d tokens <= %d limit",
            current_tokens,
            max_tokens,
        )
        return messages

    # Trim from the beginning (oldest) by removing pairs
    trimmed = list(messages)
    while trimmed and calculate_history_tokens(trimmed) > max_tokens:
        # Remove oldest pair (Human + AI) or single message
        if len(trimmed) >= 2:
            removed_human = trimmed.pop(0)
            removed_ai = trimmed.pop(0)
            logger.debug(
                "Trimmed oldest pair: human_len=%d, ai_len=%d",
                len(removed_human.content),
                len(removed_ai.content),
            )
        else:
            # Only one message left and it's still over budget
            logger.warning(
                "Single message exceeds token budget (%d tokens > %d limit). "
                "Keeping message to preserve minimal context.",
                calculate_history_tokens(trimmed),
                max_tokens,
            )
            break

    final_tokens = calculate_history_tokens(trimmed)
    logger.info(
        "History trimmed from %d to %d messages, %d tokens (limit: %d)",
        len(messages),
        len(trimmed),
        final_tokens,
        max_tokens,
    )

    return trimmed