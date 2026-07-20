"""
Hoku Health Care - RAG Pipeline Core (Day 5).

HokuRAG wraps the Hoku FAQ knowledge base stored in the `vector_store`
table (see app/models/vector_store.py) with a similarity search, so the
chatbot can ground its answers in Hoku Health Care's actual
services/policies instead of pure LLM general knowledge.

Design notes:
- pgvector is the intended production backend (project spec: PostgreSQL
  only, no Chroma/FAISS). On PostgreSQL with pgvector installed,
  similarity search uses pgvector's cosine-distance operator directly
  on the `embedding` column.
- Your current environment is SQLite with no pgvector installed, so
  similarity_search automatically falls back to computing cosine
  similarity in Python over the loaded rows. This is the path that
  actually runs for you today; the pgvector path activates automatically
  if/when DATABASE_URL points at Postgres and pgvector is installed --
  no code changes needed either way (see app/models/vector_store.py).
- Similarity threshold 0.75: chosen to balance precision/recall for
  healthcare FAQ retrieval -- high enough to avoid grounding a reply on
  a loosely related FAQ (which could read as a subtle misdiagnosis),
  low enough to still catch paraphrased patient questions.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.embeddings import EmbeddingManager
from app.core.config import settings
from app.core.database import SessionLocal, engine
from app.models.vector_store import PGVECTOR_AVAILABLE, VectorStore

logger = logging.getLogger(__name__)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Pure-Python cosine similarity -- the path that runs on SQLite."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HokuRAG:
    """
    Retrieval-Augmented Generation pipeline for Hoku Health Care FAQs.

    Usage:
        rag = HokuRAG()
        rag.create_vector_store()
        rag.add_faq_documents(faqs)
        context = rag.build_context("Do you offer home nursing in Lahore?")
    """

    def __init__(self, db: Optional[Session] = None) -> None:
        """Load the embedding model and connect to the vector store."""
        self.embedding_manager = EmbeddingManager()
        self.collection_name = settings.COLLECTION_NAME
        self.top_k = settings.RAG_TOP_K
        self.similarity_threshold = settings.RAG_SIMILARITY_THRESHOLD
        self._owns_session = db is None
        self._db = db or SessionLocal()
        self._is_postgres = engine.dialect.name == "postgresql"

    def __del__(self) -> None:
        """Close the session if HokuRAG opened it itself."""
        try:
            if self._owns_session:
                # Avoid performing session close if the logging subsystem
                # is already being torn down; SQLAlchemy emits logs when
                # rolling back/closing which can raise during interpreter
                # shutdown. If there are no active handlers, skip close.
                if logging.root and logging.root.handlers:
                    self._db.close()
                else:
                    # Handlers gone → likely interpreter shutdown; skip
                    # closing to avoid noisy tracebacks during exit.
                    pass
        except Exception:
            # Swallow any error during interpreter teardown
            pass

    # ------------------------------------------------------------------
    # Collection lifecycle
    # ------------------------------------------------------------------
    def create_vector_store(self) -> None:
        """
        Initialize the "hoku_health_faqs" collection.

        On PostgreSQL with pgvector: ensures the pgvector extension is
        enabled and the vector_store table exists (idempotent). On
        SQLite (your current setup): just ensures the table exists,
        since pgvector-specific DDL doesn't apply.
        """
        if self._is_postgres:
            try:
                from sqlalchemy import text

                self._db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                self._db.commit()
                logger.info("pgvector extension enabled (or already present)")
            except Exception as exc:
                logger.warning(
                    "Could not enable pgvector extension (may require superuser): %s", exc
                )
                self._db.rollback()

        from app.core.database import Base

        Base.metadata.create_all(bind=engine, tables=[VectorStore.__table__])
        logger.info(
            "Vector store collection '%s' initialized (postgres=%s, pgvector_available=%s)",
            self.collection_name,
            self._is_postgres,
            PGVECTOR_AVAILABLE,
        )

    def add_faq_documents(self, faqs: List[Dict[str, Any]]) -> int:
        """
        Embed and store a batch of FAQ entries.

        Args:
            faqs: List of dicts with "question", "answer", and "category" keys.

        Returns:
            int: Number of documents added.
        """
        added = 0
        for faq in faqs:
            question = faq.get("question", "")
            answer = faq.get("answer", "")
            category = faq.get("category", "general")
            content = f"Q: {question}\nA: {answer}"

            embedding = self.embedding_manager.get_embedding(content)

            row = VectorStore(
                content=content,
                embedding=embedding,
                doc_metadata={"question": question, "answer": answer, "category": category},
                category=category,
            )
            self._db.add(row)
            added += 1

        self._db.commit()
        logger.info("Added %d FAQ documents to '%s'", added, self.collection_name)
        return added

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Return the top-k most similar FAQ documents to the query.

        Uses pgvector's cosine-distance operator on PostgreSQL; falls
        back to in-Python cosine similarity over all rows on other
        backends -- this is the path that runs in your SQLite setup.
        """
        start_time = time.perf_counter()
        query_embedding = self.embedding_manager.get_embedding(query)

        results: List[Document] = []
        scores: List[float] = []

        try:
            if self._is_postgres:
                # pgvector cosine_distance: 0 = identical, 2 = opposite.
                # similarity = 1 - distance.
                stmt = (
                    select(VectorStore)
                    .order_by(VectorStore.embedding.cosine_distance(query_embedding))
                    .limit(k)
                )
                rows = list(self._db.execute(stmt).scalars().all())
                for row in rows:
                    distance = _cosine_distance_placeholder(row.embedding, query_embedding)
                    score = 1.0 - distance
                    results.append(
                        Document(
                            page_content=row.content,
                            metadata={**row.doc_metadata, "category": row.category, "score": score},
                        )
                    )
                    scores.append(score)
            else:
                all_rows = list(self._db.execute(select(VectorStore)).scalars().all())
                scored = [
                    (row, _cosine_similarity(row.embedding or [], query_embedding))
                    for row in all_rows
                ]
                scored.sort(key=lambda pair: pair[1], reverse=True)
                for row, score in scored[:k]:
                    results.append(
                        Document(
                            page_content=row.content,
                            metadata={**row.doc_metadata, "category": row.category, "score": score},
                        )
                    )
                    scores.append(score)
        except Exception as exc:
            logger.warning("similarity_search failed, returning no results: %s", exc)
            return []

        elapsed = time.perf_counter() - start_time
        logger.info(
            "RAG similarity_search: k=%d, scores=%s, latency=%.3fs",
            k,
            [round(s, 3) for s in scores],
            elapsed,
        )
        return results

    def build_context(self, query: str, threshold: Optional[float] = None) -> str:
        """
        Build a context string from the top-k FAQ answers for a query.

        If the top result's similarity is below `threshold`, returns ""
        so the chatbot falls back to general LLM knowledge rather than
        grounding on a loosely related (or irrelevant) FAQ.
        """
        effective_threshold = threshold if threshold is not None else self.similarity_threshold
        results = self.similarity_search(query, k=self.top_k)

        if not results:
            return ""

        top_score = results[0].metadata.get("score", 0.0)
        if top_score < effective_threshold:
            logger.info(
                "Top RAG score %.3f below threshold %.3f -- falling back to general knowledge",
                top_score,
                effective_threshold,
            )
            return ""

        context_parts = []
        for doc in results:
            question = doc.metadata.get("question", "")
            answer = doc.metadata.get("answer", "")
            context_parts.append(f"- Q: {question}\n  A: {answer}")

        return "\n".join(context_parts)


def _cosine_distance_placeholder(vec_a: Optional[List[float]], vec_b: List[float]) -> float:
    """
    Compute cosine distance (1 - cosine_similarity) in Python.

    Used to report a human-readable score alongside pgvector results
    without a second round-trip to the database.
    """
    if not vec_a:
        return 1.0
    return 1.0 - _cosine_similarity(vec_a, vec_b)
