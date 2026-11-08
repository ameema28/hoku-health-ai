"""
FAQ retrieval evaluation (no LLM, no running service).

Seeds the 20 FAQs into a fresh in-memory SQLite index (or, with
--database-url, into an already migrated database such as pgvector) using
the real embedding model, then runs every question in the dev set
(data/eval/retrieval_eval.jsonl) and the held-out set
(data/eval/retrieval_heldout.jsonl) through HokuRAG.similarity_search.

Per set it reports, for in-scope questions (top 3 results):
  hit@1, hit@3, MRR          rank of the expected FAQ, threshold ignored
  gated_hit@1                top-1 is the expected FAQ AND passes the threshold
and for off-topic questions:
  rejected                   top-1 score is below the threshold
  max_score                  highest top-1 score among off-topic questions

The threshold is RAG_SIMILARITY_THRESHOLD unless --threshold is given.
Choose a threshold on the dev set only, and report both sets.

Usage:
    python scripts/eval_retrieval.py --out C:/path/outside/repo/retrieval_eval.json
    python scripts/eval_retrieval.py --threshold 0.45 --out ...
    python scripts/eval_retrieval.py --sweep --out ...   # dev-set threshold sweep
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data" / "eval"
DATASETS = {
    "dev": DATA_DIR / "retrieval_eval.jsonl",
    "heldout": DATA_DIR / "retrieval_heldout.jsonl",
}
TOP_K = 3
SWEEP = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)

#: (query) -> [(faq question, score), ...] best first
Search = Callable[[str], List[Tuple[str, float]]]


def resolve_out_path(out: str) -> Path:
    """The results file must be outside the repository."""
    path = Path(out).resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"--out must be outside the repository ({REPO_ROOT})")
    return path


def load_dataset(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows.append({"query": row["query"], "expected": row.get("expected")})
    return rows


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 3) if denominator else None


def evaluate(rows: Sequence[Dict], search: Search, threshold: float) -> Dict:
    """Score one dataset. Pure function of the search results."""
    in_scope = [r for r in rows if r["expected"] is not None]
    off_topic = [r for r in rows if r["expected"] is None]

    details = []
    hit1 = hit3 = gated = 0
    reciprocal = 0.0
    for row in in_scope:
        hits = search(row["query"])[:TOP_K]
        names = [name for name, _ in hits]
        rank = names.index(row["expected"]) + 1 if row["expected"] in names else None
        top_score = hits[0][1] if hits else 0.0
        hit1 += rank == 1
        hit3 += rank is not None
        reciprocal += 1.0 / rank if rank else 0.0
        gated += rank == 1 and top_score >= threshold
        details.append(
            {
                "query": row["query"],
                "expected": row["expected"],
                "rank": rank,
                "top1": names[0] if names else None,
                "top1_score": round(top_score, 3),
            }
        )

    rejected = 0
    off_scores = []
    for row in off_topic:
        hits = search(row["query"])[:1]
        top_score = hits[0][1] if hits else 0.0
        off_scores.append(top_score)
        rejected += top_score < threshold
        details.append(
            {
                "query": row["query"],
                "expected": None,
                "top1": hits[0][0] if hits else None,
                "top1_score": round(top_score, 3),
                "passes_threshold": top_score >= threshold,
            }
        )

    n = len(in_scope)
    return {
        "threshold": threshold,
        "in_scope": n,
        "off_topic": len(off_topic),
        "hit_at_1": _rate(hit1, n),
        "hit_at_3": _rate(hit3, n),
        "mrr": round(reciprocal / n, 3) if n else None,
        "gated_hit_at_1": _rate(gated, n),
        "off_topic_rejected": rejected,
        "off_topic_rejection_rate": _rate(rejected, len(off_topic)),
        "off_topic_max_score": round(max(off_scores), 3) if off_scores else None,
        "rows": details,
    }


def build_search(database_url: Optional[str] = None) -> Search:
    """Seed the FAQs and return a search function over the real model."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.ai.rag import HokuRAG
    from app.scripts.seed_faqs import FAQS

    if database_url:
        engine = create_engine(database_url)
    else:
        from sqlalchemy.pool import StaticPool

        from app.core.database import Base
        import app.models  # noqa: F401  (registers the tables)

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)

    session = sessionmaker(bind=engine)()
    rag = HokuRAG(db=session)
    rag.upsert_faq_documents(FAQS)
    if not any(rag.embedding_manager.get_embedding("probe")):
        raise RuntimeError("embedding model unavailable (zero-vector fallback)")

    def search(query: str) -> List[Tuple[str, float]]:
        return [
            (doc.metadata.get("question", ""), float(doc.metadata.get("score", 0.0)))
            for doc in rag.similarity_search(query, k=TOP_K)
        ]

    return search


def _cached(search: Search) -> Search:
    cache: Dict[str, List[Tuple[str, float]]] = {}

    def wrapped(query: str) -> List[Tuple[str, float]]:
        if query not in cache:
            cache[query] = search(query)
        return cache[query]

    return wrapped


def _print_set(name: str, result: Dict) -> None:
    print(
        f"{name:8s} thr={result['threshold']:.2f}  in-scope={result['in_scope']}"
        f"  hit@1={result['hit_at_1']}  hit@3={result['hit_at_3']}"
        f"  MRR={result['mrr']}  gated_hit@1={result['gated_hit_at_1']}"
        f"  off-topic rejected={result['off_topic_rejected']}/{result['off_topic']}"
        f"  off-topic max={result['off_topic_max_score']}"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", required=True, help="JSON results file, outside the repository"
    )
    parser.add_argument(
        "--threshold", type=float, help="Override RAG_SIMILARITY_THRESHOLD"
    )
    parser.add_argument(
        "--sweep", action="store_true", help="Also sweep thresholds on the DEV set"
    )
    parser.add_argument(
        "--database-url", help="Use this migrated database instead of SQLite"
    )
    args = parser.parse_args(argv)
    out = resolve_out_path(args.out)

    from app.core.config import settings

    threshold = (
        args.threshold
        if args.threshold is not None
        else settings.RAG_SIMILARITY_THRESHOLD
    )
    search = _cached(build_search(args.database_url))
    datasets = {name: load_dataset(path) for name, path in DATASETS.items()}

    results: Dict = {
        "backend": "postgres" if args.database_url else "sqlite",
        "top_k": TOP_K,
        "sets": {},
    }
    for name, rows in datasets.items():
        results["sets"][name] = evaluate(rows, search, threshold)
        _print_set(name, results["sets"][name])

    if args.sweep:
        results["dev_sweep"] = []
        for value in SWEEP:
            res = evaluate(datasets["dev"], search, value)
            res.pop("rows")
            results["dev_sweep"].append(res)
            _print_set("dev", res)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
