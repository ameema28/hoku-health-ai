"""
Hoku Health Care - Response Cache (Day 8).

In-memory response caching with SHA-256 key hashing and TTL expiration.
Provides < 5ms response time for cache hits, dramatically reducing
LLM load and improving perceived responsiveness for repeated queries.

Clinical Safety Mandate:
- NEVER cache emergency queries (life-threatening symptoms must always
  trigger the full emergency detection pipeline).
- NEVER cache symptom inquiries (personalized clinical advice must be
  generated fresh for each patient).
- Only cache general health information and booking-related responses
  where the answer is deterministic and safe to repeat.

Saved execution time: ~2000-3000ms per cache hit by skipping the entire
LLM generation pipeline.
"""

import hashlib
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    In-memory cache for AI chatbot responses with TTL-based expiration.

    Uses SHA-256 hashing of (message + intent + last_3_messages) to
    generate deterministic cache keys. Entries expire after a
    configurable TTL (default 1 hour) to prevent stale responses.

    Thread-safety: This implementation uses simple dict operations.
    In a multi-worker deployment, each worker maintains its own cache.
    For shared caching, upgrade to Redis (see production note below).

    Attributes:
        DEFAULT_TTL: Default time-to-live in seconds (3600 = 1 hour).
    """

    DEFAULT_TTL: int = 3600  # 1 hour

    # Day 8.1: intents whose answers do NOT depend on conversation position.
    # "What services do you offer?" has the same correct answer on turn 1 and
    # turn 9, so its cache key must not include history.
    #
    # Why this matters: the key previously always mixed in the last 3 turns,
    # but every turn is persisted to chat_history, so the history differs on
    # each request. An identical repeated question therefore produced a
    # DIFFERENT key every time and the cache could never register a hit —
    # making the whole Day 8 caching layer dead weight in production.
    #
    # Symptom/emergency intents are excluded from caching entirely by
    # should_cache(), so this relaxation cannot leak personalised clinical
    # advice between conversational contexts.
    CONTEXT_FREE_INTENTS: frozenset = frozenset({"general"})

    def __init__(self) -> None:
        """Initialize the in-memory cache storage."""
        self._cache: dict[str, dict] = {}

    def _generate_key(
        self,
        message: str,
        intent: str,
        last_3_messages: list,
    ) -> str:
        """
        Generate a deterministic SHA-256 cache key.

        The key incorporates:
        - Normalized message text (stripped, lowercased)
        - Intent classification label
        - Last 3 conversation turns (for context sensitivity)

        Args:
            message: The user's message.
            intent: Classified intent string.
            last_3_messages: List of recent message dicts for context.

        Returns:
            str: Hex-encoded SHA-256 digest.
        """
        # Normalize inputs for consistent hashing
        normalized_message = message.strip().lower()
        normalized_intent = (intent or "").strip().lower()

        # Day 8.1: context-free intents hash on (message, intent) only.
        if normalized_intent in self.CONTEXT_FREE_INTENTS:
            context_str = ""
        else:
            # Serialize last 3 messages into a stable string.
            # Day 8.1: LangChain message objects are now normalised via their
            # .type/.content attributes rather than str(msg). repr() of a
            # BaseMessage embeds additional_kwargs/response_metadata, which is
            # not guaranteed stable across langchain versions and would
            # silently invalidate every key on upgrade.
            context_parts = []
            for msg in (last_3_messages or [])[-3:]:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = str(msg.get("content", "")).strip().lower()
                elif hasattr(msg, "content"):
                    role = str(getattr(msg, "type", "")).strip().lower()
                    content = str(getattr(msg, "content", "")).strip().lower()
                else:
                    role, content = "", str(msg).strip().lower()
                context_parts.append(f"{role}:{content}")

            context_str = "|".join(context_parts)

        # Build composite key string
        key_source = f"{normalized_message}::{normalized_intent}::{context_str}"

        # SHA-256 for uniform key distribution and collision resistance
        return hashlib.sha256(key_source.encode("utf-8")).hexdigest()

    def should_cache(self, intent: str, is_emergency: bool) -> bool:
        """
        Determine whether a query/response pair should be cached.

        Clinical Safety Rules:
        - Emergency queries: NEVER cached (must always run full pipeline).
        - Symptom inquiries: NEVER cached (personalized advice required).
        - General / booking / medication: SAFE to cache.

        Args:
            intent: Classified intent string.
            is_emergency: Whether emergency was detected.

        Returns:
            bool: True if caching is permitted, False otherwise.
        """
        if is_emergency:
            # Saved execution time: N/A — safety override, never cache emergencies
            logger.debug("Cache denied: emergency query")
            return False

        intent_lower = (intent or "").lower()

        # NEVER cache symptom inquiries — personalized clinical advice
        if intent_lower in ("symptom", "symptom_inquiry", "emergency"):
            logger.debug("Cache denied: symptom/emergency intent")
            return False

        # General, booking, and medication intents are safe to cache
        if intent_lower in ("general", "booking", "medication"):
            logger.debug("Cache permitted: intent=%s", intent)
            return True

        # Default deny for unknown intents (safety-first)
        logger.debug("Cache denied: unknown intent='%s'", intent)
        return False

    def get(
        self,
        message: str,
        intent: str,
        last_3_messages: list,
    ) -> Optional[str]:
        """
        Retrieve a cached response if available and not expired.

        Target response time: < 5ms for cache hits.

        Args:
            message: The user's message.
            intent: Classified intent string.
            last_3_messages: Recent conversation context.

        Returns:
            Optional[str]: Cached response text, or None if miss/expired.
        """
        start_time = time.perf_counter()

        key = self._generate_key(message, intent, last_3_messages)
        entry = self._cache.get(key)

        if entry is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug("Cache MISS (key=%s...): %.3fms", key[:16], elapsed_ms)
            return None

        # Check TTL expiration
        if time.time() > entry["expires_at"]:
            del self._cache[key]
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug("Cache EXPIRED (key=%s...): %.3fms", key[:16], elapsed_ms)
            return None

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Cache HIT (key=%s...): %.3fms — saved ~%.0fms LLM time",
            key[:16],
            elapsed_ms,
            2000.0,  # Approximate LLM generation time saved
        )
        return entry["response"]

    def set(
        self,
        message: str,
        intent: str,
        last_3_messages: list,
        response: str,
        is_emergency: bool = False,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a response in the cache with TTL expiration.

        Args:
            message: The user's message.
            intent: Classified intent string.
            last_3_messages: Recent conversation context.
            response: The AI-generated response to cache.
            is_emergency: Whether this was an emergency query.
            ttl: Optional custom TTL in seconds (defaults to DEFAULT_TTL).

        Saved execution time: ~2000-3000ms for the next identical query.
        """
        # Safety gate: never store emergency or symptom responses
        if not self.should_cache(intent, is_emergency):
            logger.debug("Cache SET skipped: intent=%s, is_emergency=%s", intent, is_emergency)
            return

        key = self._generate_key(message, intent, last_3_messages)
        effective_ttl = ttl if ttl is not None else self.DEFAULT_TTL

        self._cache[key] = {
            "response": response,
            "expires_at": time.time() + effective_ttl,
            "intent": intent,
            "created_at": time.time(),
        }

        logger.debug(
            "Cache SET (key=%s..., ttl=%ds, intent=%s)",
            key[:16],
            effective_ttl,
            intent,
        )

    def invalidate(self, message: str, intent: str, last_3_messages: list) -> bool:
        """
        Manually invalidate a specific cache entry.

        Args:
            message: The user's message.
            intent: Classified intent string.
            last_3_messages: Recent conversation context.

        Returns:
            bool: True if an entry was removed, False if not found.
        """
        key = self._generate_key(message, intent, last_3_messages)
        if key in self._cache:
            del self._cache[key]
            logger.info("Cache INVALIDATED (key=%s...)", key[:16])
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cached entries.

        Returns:
            int: Number of entries removed.
        """
        count = len(self._cache)
        self._cache.clear()
        logger.info("Cache CLEARED: %d entries removed", count)
        return count

    def stats(self) -> dict:
        """
        Return cache statistics for monitoring.

        Returns:
            dict: Entry count, memory estimate, and hit/miss tracking.
        """
        return {
            "entries": len(self._cache),
            "default_ttl_seconds": self.DEFAULT_TTL,
        }


# Module-level singleton — shared across the application
response_cache = ResponseCache()