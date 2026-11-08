"""
Retrieval evaluation: metric math, the output-path guard, and regression
floors on the dev and held-out sets with the real embedding model (SQLite).

Floors are the values measured on 2026-11-09 at RAG_SIMILARITY_THRESHOLD=0.35
rounded down to a multiple of 0.05 (dev: hit@1 0.938, hit@3 1.0, MRR 0.964,
gated hit@1 0.875, off-topic 8/8; held-out: hit@1 0.90, hit@3 0.90, MRR 0.90,
gated hit@1 0.90, off-topic 10/12).
"""

import pytest

from scripts import eval_retrieval as ev

FLOORS = {
    "dev": {
        "hit_at_1": 0.90,
        "hit_at_3": 1.00,
        "mrr": 0.95,
        "gated_hit_at_1": 0.85,
        "off_topic_rejection_rate": 1.00,
    },
    "heldout": {
        "hit_at_1": 0.90,
        "hit_at_3": 0.90,
        "mrr": 0.90,
        "gated_hit_at_1": 0.90,
        "off_topic_rejection_rate": 0.80,
    },
}


def test_metric_math_on_fake_search():
    results = {
        "a": [("A", 0.9), ("B", 0.5)],
        "b": [("X", 0.8), ("B", 0.7)],
        "c": [("C", 0.2)],
        "off1": [("A", 0.6)],
        "off2": [("A", 0.1)],
    }
    rows = [
        {"query": "a", "expected": "A"},
        {"query": "b", "expected": "B"},
        {"query": "c", "expected": "C"},
        {"query": "off1", "expected": None},
        {"query": "off2", "expected": None},
    ]
    res = ev.evaluate(rows, lambda q: results[q], threshold=0.35)
    assert res["hit_at_1"] == round(2 / 3, 3)
    assert res["hit_at_3"] == 1.0
    assert res["mrr"] == round((1 + 0.5 + 1) / 3, 3)
    assert res["gated_hit_at_1"] == round(1 / 3, 3)  # "c" is below threshold
    assert res["off_topic_rejected"] == 1
    assert res["off_topic_max_score"] == 0.6


def test_out_path_must_be_outside_repo(tmp_path):
    with pytest.raises(ValueError):
        ev.resolve_out_path(str(ev.REPO_ROOT / "results.json"))
    assert ev.resolve_out_path(str(tmp_path / "r.json")) == (tmp_path / "r.json")


def test_datasets_have_required_sizes():
    dev = ev.load_dataset(ev.DATASETS["dev"])
    held = ev.load_dataset(ev.DATASETS["heldout"])
    assert sum(r["expected"] is not None for r in dev) >= 25
    assert sum(r["expected"] is None for r in dev) >= 5
    assert sum(r["expected"] is not None for r in held) >= 15
    assert sum(r["expected"] is None for r in held) >= 10
    assert not {r["query"] for r in dev} & {r["query"] for r in held}


@pytest.fixture(scope="module")
def search():
    return ev._cached(ev.build_search())


@pytest.mark.parametrize("name", ["dev", "heldout"])
def test_regression_floors(search, name):
    from app.core.config import settings

    res = ev.evaluate(
        ev.load_dataset(ev.DATASETS[name]), search, settings.RAG_SIMILARITY_THRESHOLD
    )
    for metric, floor in FLOORS[name].items():
        assert res[metric] >= floor, (metric, res[metric], floor)
