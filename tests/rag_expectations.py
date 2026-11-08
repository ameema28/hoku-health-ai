"""Expected top-1 FAQ per paraphrased query (all-MiniLM-L6-v2, the 20 seeded FAQs).

Settled by running the queries (2026-11-08, pgvector and in-Python cosine gave
the same top-1 and the same scores to 4 decimals), not assumed.
"""

EXPECTED_TOP1 = {
    "what are your opening hours": "What are Hoku Health Care's operating hours?",
    "nursing care at home after an operation": (
        "What post-surgical care does Hoku provide at home?"
    ),
    "can a nurse give me my insulin shot at home": (
        "Can Hoku nurses administer prescribed injections at home?"
    ),
}
