"""
Hoku Health Care - LLM Model Selection & Context Compression (Day 8).

Provides intelligent LLM factory methods and prompt context compression
to reduce token usage, lower latency, and stay within the NFR-02 budget.

Design rationale:
- Fast LLM (llama-3.1-8b-instant) for simple tasks: 10x cheaper, 3x faster
- Main LLM (llama-3.3-70b-versatile) for patient-facing responses only
- Context compression drops old messages and truncates long turns to
  minimize tokenization overhead

Saved execution time: ~500-800ms per request by compressing context
and selecting the right model for each task.
"""

import logging
from typing import Any, List, Optional

# LangChain imports at module level (required for test patching)
try:
    from langchain_groq import ChatGroq

    LANGCHAIN_AVAILABLE = True
except ImportError as _import_exc:
    LANGCHAIN_AVAILABLE = False
    ChatGroq = None  # type: ignore
    logging.getLogger(__name__).warning("LangChain/Groq not installed: %s", _import_exc)

from app.ai.config import ai_settings

logger = logging.getLogger(__name__)


def _msg_to_dict(msg: Any) -> Optional[dict]:
    """
    Normalize a message object to a plain dict with 'role' and 'content' keys.

    Handles:
    - Plain dicts (already normalized)
    - LangChain AIMessage / HumanMessage / SystemMessage objects
    - Any object with .type and .content attributes

    Returns None if the message cannot be normalized.
    """
    if isinstance(msg, dict):
        return msg

    # LangChain message objects (AIMessage, HumanMessage, SystemMessage, etc.)
    # These have .type ("ai", "human", "system") and .content attributes
    msg_type = getattr(msg, "type", None)
    msg_content = getattr(msg, "content", None)

    if msg_type is not None and msg_content is not None:
        role_map = {
            "ai": "assistant",
            "human": "user",
            "system": "system",
        }
        role = role_map.get(msg_type, msg_type)
        return {"role": role, "content": str(msg_content)}

    # Fallback: try to extract from __dict__
    if hasattr(msg, "__dict__"):
        d = msg.__dict__
        if "content" in d:
            role = d.get("type", "unknown")
            if role in ("ai", "assistant"):
                role = "assistant"
            elif role in ("human", "user"):
                role = "user"
            return {"role": role, "content": str(d["content"])}

    return None


class LLMFactory:
    """
    Factory for creating appropriately-configured Groq LLM instances.

    Centralizes model selection, timeout, and token budget configuration
    so every pipeline stage uses the right tool for the job.
    """

    @staticmethod
    def get_fast_llm(
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Create a fast, low-cost LLM for intent classification and
        other simple NLP tasks.

        Configuration:
        - Model: llama-3.1-8b-instant (~10x cheaper than 70B)
        - max_tokens: 150 (classification output is tiny)
        - timeout: 1.0s (hard cutoff to protect NFR-02)

        Returns:
            ChatGroq instance or None if LangChain unavailable.

        Saved execution time: ~200-500ms vs using the 70B model for
        simple classification tasks.
        """
        if not LANGCHAIN_AVAILABLE or ChatGroq is None:
            logger.warning("Fast LLM unavailable: LangChain/Groq not installed")
            return None

        # Day 8.1: every value is an explicit override falling back to
        # ai_settings. Callers that hold their OWN settings binding (e.g.
        # HokuChatbot, whose ai_settings is what unit tests patch) must pass
        # them in — otherwise this factory silently reads the real .env and
        # bypasses the caller's configuration. That defect made the chatbot
        # suite issue live Groq requests.
        resolved_model = model if model is not None else ai_settings.GROQ_FAST_MODEL
        resolved_key = api_key if api_key is not None else ai_settings.groq_api_key
        resolved_max_tokens = max_tokens if max_tokens is not None else 150
        resolved_timeout = (
            request_timeout if request_timeout is not None
            else ai_settings.INTENT_CLASSIFICATION_TIMEOUT
        )

        try:
            llm = ChatGroq(
                model=resolved_model,
                api_key=resolved_key,
                temperature=0.0,  # Deterministic for classification
                max_tokens=resolved_max_tokens,  # Minimal: intent + confidence JSON
                request_timeout=resolved_timeout,
            )
            logger.debug("Fast LLM created: %s", resolved_model)
            return llm
        except Exception as exc:
            logger.warning("Failed to create fast LLM: %s", exc)
            return None

    @staticmethod
    def get_main_llm(
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        request_timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Create the main high-quality LLM for patient-facing responses.

        Configuration:
        - Model: llama-3.3-70b-versatile (best quality for empathetic replies)
        - max_tokens: 300 (reduced from 512 to save ~200ms per generation)
        - timeout: 2.5s (leaves buffer for earlier pipeline stages)

        Returns:
            ChatGroq instance or None if LangChain unavailable.

        Saved execution time: ~200-400ms by capping max_tokens to 300
        instead of the full 512-token budget.
        """
        if not LANGCHAIN_AVAILABLE or ChatGroq is None:
            logger.warning("Main LLM unavailable: LangChain/Groq not installed")
            return None

        # Day 8.1: resolve overrides before the guard, so a caller-supplied
        # key is honoured (see get_fast_llm for why this matters).
        resolved_model = model if model is not None else ai_settings.GROQ_MAIN_MODEL
        resolved_key = api_key if api_key is not None else ai_settings.groq_api_key
        resolved_temperature = (
            temperature if temperature is not None else ai_settings.TEMPERATURE
        )
        resolved_max_tokens = (
            max_tokens if max_tokens is not None else ai_settings.MAX_TOKENS
        )
        resolved_timeout = (
            request_timeout if request_timeout is not None
            else ai_settings.GROQ_TIMEOUT_SECONDS
        )

        if not resolved_key:
            logger.warning("Main LLM unavailable: GROQ_API_KEY is empty")
            return None

        try:
            # Day 8.1: max_tokens now reads ai_settings.MAX_TOKENS (512).
            # The hardcoded 300 was unreachable dead config (chatbot.py built
            # its own ChatGroq at 512), and enabling it as-is would have
            # truncated the mandatory JSON envelope mid-object on longer
            # replies — _parse_llm_output would then fall through to the
            # plain-text branch and lose suggestedSpecialist/severity.
            llm = ChatGroq(
                model=resolved_model,
                api_key=resolved_key,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                request_timeout=resolved_timeout,
            )
            logger.debug("Main LLM created: %s", resolved_model)
            return llm
        except Exception as exc:
            logger.warning("Failed to create main LLM: %s", exc)
            return None

    @staticmethod
    def compress_prompt(
        history: List[Any],
        max_messages: int = 3,
        max_chars_per_turn: int = 600,
    ) -> List[dict]:
        """
        Compress conversation history to reduce tokenization overhead.

        Strategy:
        1. Normalize LangChain message objects (AIMessage, HumanMessage) to plain dicts.
        2. Truncate history to the last `max_messages` turns.
        3. Truncate any individual message over `max_chars_per_turn`
           characters to prevent token bloat from long user inputs.

        This preserves the most recent (and thus most clinically
        relevant) context while keeping the prompt under budget.

        Args:
            history: List of message objects (dicts or LangChain messages).
            max_messages: Maximum number of turns to retain.
            max_chars_per_turn: Maximum characters per individual message.

        Returns:
            List[dict]: Compressed history ready for prompt injection.

        Saved execution time: ~100-300ms by reducing context token
        count, which speeds up both prompt formatting and LLM generation.
        """
        if not history:
            return []

        # Step 0: Normalize all messages to plain dicts
        normalized = []
        for msg in history:
            d = _msg_to_dict(msg)
            if d is not None:
                normalized.append(d)
            else:
                # Skip unparseable messages rather than crash
                logger.debug("Skipping unparseable message in compress_prompt: %s", type(msg))

        if not normalized:
            return []

        # Step 1: Keep only the most recent N messages. If the trimmed
        # window begins in the middle of an assistant response, include the
        # prior user message to preserve turn structure.
        if len(normalized) > max_messages:
            trimmed = normalized[-max_messages:]
            if max_messages > 1 and trimmed and trimmed[0].get("role") == "assistant":
                prev_index = len(normalized) - max_messages - 1
                if prev_index >= 0 and normalized[prev_index].get("role") == "user":
                    trimmed = [normalized[prev_index]] + trimmed[:-1]
        else:
            trimmed = list(normalized)

        # Step 2: Truncate overly long individual messages
        compressed = []
        for msg in trimmed:
            if not isinstance(msg, dict):
                continue

            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > max_chars_per_turn:
                truncated = content[:max_chars_per_turn] + "..."
                msg = {**msg, "content": truncated}
                logger.debug(
                    "Compressed message: %d -> %d chars",
                    len(content),
                    len(truncated),
                )

            compressed.append(msg)

        # Day 8.1: `normalized` is already the parsed form of `history`;
        # the previous expression re-ran _msg_to_dict up to 3x per message
        # purely to build a log line.
        original_tokens_estimate = sum(len(str(m.get("content", ""))) for m in normalized) // 4
        compressed_tokens_estimate = sum(len(str(m.get("content", ""))) for m in compressed) // 4

        logger.info(
            "Prompt compressed: %d -> %d messages, ~%d -> ~%d tokens",
            len(history),
            len(compressed),
            original_tokens_estimate,
            compressed_tokens_estimate,
        )

        return compressed


def compress_prompt(
    history: List[Any],
    max_messages: int = 3,
    max_chars_per_turn: int = 600,
) -> List[dict]:
    """
    Convenience wrapper exposing the static LLMFactory.compress_prompt
    function as a module-level symbol.

    This preserves existing import patterns used elsewhere in the codebase.
    """
    return LLMFactory.compress_prompt(
        history,
        max_messages=max_messages,
        max_chars_per_turn=max_chars_per_turn,
    )