"""Retrieval metrics — extend with nDCG, recall@k, etc."""

from __future__ import annotations

from core.types import ScoredChunk
from eval.dataset import EvalExample


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


def evaluate_examples(
    examples: list[EvalExample],
    retrieve_fn,
    *,
    top_k: int = 5,
) -> dict[str, float]:
    hits: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []

    for example in examples:
        retrieved = retrieve_fn(example.query, top_k=top_k)
        hits.append(hit_at_k(retrieved, example.expected_doc_ids, k=top_k))
        recalls.append(recall_at_k(retrieved, example.expected_doc_ids, k=top_k))
        mrrs.append(mrr(retrieved, example.expected_doc_ids))

    n = len(examples) or 1
    return {
        f"hit@{top_k}": sum(hits) / n,
        f"recall@{top_k}": sum(recalls) / n,
        "mrr": sum(mrrs) / n,
        "count": float(len(examples)),
    }


def evaluate_examples_by_language(
    examples: list[EvalExample],
    retrieve_fn,
    *,
    top_k: int = 5,
) -> dict[str, dict[str, float]]:
    by_language: dict[str, list[EvalExample]] = {}
    for example in examples:
        by_language.setdefault(example.language, []).append(example)

    return {
        language: evaluate_examples(lang_examples, retrieve_fn, top_k=top_k)
        for language, lang_examples in sorted(by_language.items())
    }
