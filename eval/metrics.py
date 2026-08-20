"""Retrieval metrics and the aggregation shared by every eval harness.

Scoring is deliberately two-phase: :func:`eval.significance.score_examples`
keeps one row per query, and these aggregate those rows. The per-query rows
are what the paired bootstrap resamples, so nothing here may collapse them
early.
"""

from __future__ import annotations

from core.types import ScoredChunk


def hit_at_k(retrieved: list[ScoredChunk], expected_doc_ids: list[str], k: int = 5) -> float:
    if not expected_doc_ids:
        return 0.0
    top = retrieved[:k]
    retrieved_ids = {chunk.chunk.document_id for chunk in top}
    return 1.0 if retrieved_ids.intersection(expected_doc_ids) else 0.0


def recall_at_k(retrieved: list[ScoredChunk], expected_doc_ids: list[str], k: int = 5) -> float:
    if not expected_doc_ids:
        return 0.0
    top = retrieved[:k]
    retrieved_ids = {chunk.chunk.document_id for chunk in top}
    relevant_retrieved = len(retrieved_ids.intersection(expected_doc_ids))
    return relevant_retrieved / len(expected_doc_ids)


def mrr(retrieved: list[ScoredChunk], expected_doc_ids: list[str]) -> float:
    if not expected_doc_ids:
        return 0.0
    expected = set(expected_doc_ids)
    for rank, scored in enumerate(retrieved, start=1):
        if scored.chunk.document_id in expected:
            return 1.0 / rank
    return 0.0


def aggregate_scores(
    rows: list[dict[str, float | str]],
    *,
    top_k: int = 5,
) -> dict[str, float]:
    """Aggregate per-example score rows into hit/recall/mrr means."""
    if not rows:
        return {f"hit@{top_k}": 0.0, f"recall@{top_k}": 0.0, "mrr": 0.0, "count": 0.0}

    n = len(rows)
    return {
        f"hit@{top_k}": sum(float(r["hit"]) for r in rows) / n,
        f"recall@{top_k}": sum(float(r["recall"]) for r in rows) / n,
        "mrr": sum(float(r["mrr"]) for r in rows) / n,
        "count": float(n),
    }


def aggregate_scores_by_language(
    rows: list[dict[str, float | str]],
    *,
    top_k: int = 5,
) -> dict[str, dict[str, float]]:
    by_language: dict[str, list[dict[str, float | str]]] = {}
    for row in rows:
        by_language.setdefault(str(row["language"]), []).append(row)
    return {
        language: aggregate_scores(lang_rows, top_k=top_k)
        for language, lang_rows in sorted(by_language.items())
    }
