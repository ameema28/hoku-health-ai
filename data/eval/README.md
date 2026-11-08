# Retrieval evaluation sets

Each line is a JSON object: `query` (a visitor's question) and `expected`, which is the seeded FAQ question (`app/scripts/seed_faqs.py`) that should be retrieved first, or `null` for an off-topic question that should not pass the similarity threshold.

| File | In-scope | Off-topic | Use |
|:---|:---|:---|:---|
| `retrieval_heldout.jsonl` | 20 (one per FAQ) | 12 (6 near-topic: they mention cities, bookings, delivery, cancelling or opening hours) | **Held-out.** Written and committed before any retrieval measurement. Never edited after the first measurement, and never used to choose the threshold. |
| `retrieval_eval.jsonl` | dev set | dev set | Used to inspect results and to choose the threshold. |

Rules:
- The threshold (`RAG_SIMILARITY_THRESHOLD`) may only be chosen on the dev set, and any change is reported with results on **both** sets at the old and the new value.
- The paraphrases avoid copying the FAQ wording, so the sets test retrieval of reworded questions rather than exact matches.
- The two demo-page FAQ examples were checked against retrieval in `tests/test_demo_page.py` before this file was written; neither of them is in either set.

Results are written by `scripts/eval_retrieval.py` to a file outside the repository.
