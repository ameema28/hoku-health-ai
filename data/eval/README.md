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

## Results (measured 2026-11-09, SQLite path, all-MiniLM-L6-v2, top-3)

| Set | Threshold | hit@1 | hit@3 | MRR | Gated hit@1 | Off-topic rejected | Off-topic max score |
|:---|:---|:---|:---|:---|:---|:---|:---|
| dev | 0.35 | 0.938 | 1.000 | 0.964 | 0.875 | 8/8 | 0.301 |
| held-out | 0.35 | 0.900 | 0.900 | 0.900 | 0.900 | 10/12 | 0.468 |

**Threshold: unchanged at 0.35.** On the dev sweep (0.25 to 0.60) it is the lowest value that still rejects all 8 dev off-topic questions; 0.30 lets one through (gated hit@1 0.906), 0.40 loses one in-scope answer (0.844).

**Known misses on the held-out set:** "Which kinds of care can Hoku send to my house?" (top-1: elderly care FAQ, services FAQ not in the top 3) and "Can a nurse come late at night?" (operating-hours FAQ not in the top 3).

**Known leaks on the held-out set** (off-topic questions that pass 0.35): "What are the opening hours of Lahore's Emporium mall?" (0.468, matched the home-visit wait-time FAQ) and "Does Amazon deliver to Lahore?" (0.363, matched the prescription-delivery FAQ). Questions that name a city, opening hours or delivery score like real Hoku questions, so the threshold alone cannot separate them. The reply prompt's scope rule (`app/ai/prompts.py`, `REPLY_RULES`) asks the model to decline such questions; whether the live model does so is **not measured** (see the scope probes in `scripts/smoke_check.py`). Without an LLM, the fallback reply would show the matched FAQ answer.

Regression floors (the values above rounded down to 0.05) are in `tests/test_retrieval_eval.py`.
