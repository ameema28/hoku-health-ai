"""
Hoku Health Care - Embedding Manager (Day 5).

Wraps sentence-transformers/all-MiniLM-L6-v2 for local, free embedding
generation used by the pgvector FAQ RAG pipeline.

Why this model:
- Local inference, no API key required (Groq has no embeddings endpoint).
- 384 dimensions: fast and memory-efficient for a vector index.
- MIT license, works fully offline.
- Sufficient quality for short-form FAQ semantic search.

Settings note: EMBEDDING_MODEL and VECTOR_DIMENSION live on the core
app.core.config.Settings (alongside your existing stubbed
EMBEDDING_MODEL field), not on app.ai.config.AISettings -- this module
reads from app.core.config to match that existing pattern.

Windows / environment fallback:
- If the sentence-transformers import fails (e.g. missing PyTorch build
  on some Windows setups), we log a WARNING and return zero-vectors
  instead of crashing on import. This keeps the rest of the app
  importable and functional -- RAG lookups will simply never match
  above the similarity threshold, so the chatbot falls back to its
  default (non-RAG) behavior.
"""

import logging
from typing import Any, List, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as _import_exc:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore
    logger.warning(
        "sentence-transformers not available (%s). "
        "Embedding calls will return zero-vectors as a safe fallback.",
        _import_exc,
    )


class EmbeddingManager:
    """Loads the embedding model once and generates embeddings on demand."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.dimension = settings.VECTOR_DIMENSION
        self._model: Optional[Any] = None

    @property
    def model(self) -> Optional[Any]:
        """Lazily load the sentence-transformers model."""
        if self._model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model '%s' loaded", self.model_name)
            except Exception as exc:
                logger.warning("Failed to load embedding model '%s': %s", self.model_name, exc)
                self._model = None
        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate a single embedding vector for the given text.

        Returns a zero-vector of the configured dimension if the model
        is unavailable, rather than raising.
        """
        if not text or not isinstance(text, str):
            return [0.0] * self.dimension

        if self.model is None:
            logger.warning("Embedding model unavailable, returning zero-vector fallback")
            return [0.0] * self.dimension

        try:
            vector = self.model.encode(text, normalize_embeddings=True)
            return vector.tolist()
        except Exception as exc:
            logger.warning("Embedding generation failed: %s", exc)
            return [0.0] * self.dimension

    def batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts in one model call.

        Falls back to per-text zero-vectors if the model is unavailable
        or batch encoding fails.
        """
        if not texts:
            return []

        if self.model is None:
            logger.warning("Embedding model unavailable, returning zero-vector fallbacks")
            return [[0.0] * self.dimension for _ in texts]

        try:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vectors]
        except Exception as exc:
            logger.warning("Batch embedding generation failed: %s", exc)
            return [[0.0] * self.dimension for _ in texts]


# Module-level singleton -- the model is expensive to load, so we share
# one instance across the app (mirrors the HokuChatbot lazy-singleton pattern).
embedding_manager = EmbeddingManager()


async def get_embedding(text: str) -> List[float]:
    """Async-safe wrapper: runs model inference in a thread pool."""
    import asyncio

    return await asyncio.to_thread(embedding_manager.get_embedding, text)


async def batch_embed(texts: List[str]) -> List[List[float]]:
    """Async-safe wrapper: runs batch model inference in a thread pool."""
    import asyncio

    return await asyncio.to_thread(embedding_manager.batch_embed, texts)
