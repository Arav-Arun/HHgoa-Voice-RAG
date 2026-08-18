"""Bootstrap significance tests for per-query retrieval metrics."""

from __future__ import annotations

import random
from typing import Literal

from eval.dataset import EvalExample

MetricName = Literal["hit", "recall", "mrr"]


def score_examples(
    examples: list[EvalExample],
    retrieve_fn,
    *,
    top_k: int = 5,
) -> list[dict[str, float | str]]:
    from eval.metrics import hit_at_k, mrr, recall_at_k

    rows: list[dict[str, float | str]] = []
    for example in examples:
        retrieved = retrieve_fn(example.query, top_k=top_k)
        rows.append(
            {
                "language": example.language,
                "hit": hit_at_k(retrieved, example.expected_doc_ids, k=top_k),
                "recall": recall_at_k(retrieved, example.expected_doc_ids, k=top_k),
                "mrr": mrr(retrieved, example.expected_doc_ids),
            }
        )
    return rows


def bootstrap_mean_diff(
    scores_a: list[float],
    scores_b: list[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap paired comparison: mean(a) - mean(b) with two-sided p-value."""
    if len(scores_a) != len(scores_b):
        raise ValueError("score lists must be the same length for paired bootstrap")
    if not scores_a:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_value": 1.0,
            "n": 0.0,
        }

    n = len(scores_a)
    observed = sum(scores_a) / n - sum(scores_b) / n
    rng = random.Random(seed)
    diffs: list[float] = []

    for _ in range(n_resamples):
        sample = [rng.randrange(n) for _ in range(n)]
        mean_a = sum(scores_a[i] for i in sample) / n
        mean_b = sum(scores_b[i] for i in sample) / n
        diffs.append(mean_a - mean_b)

    diffs.sort()
    ci_low = diffs[int(0.025 * n_resamples)]
    ci_high = diffs[int(0.975 * n_resamples)]
    # Two-sided p: fraction of bootstrap diffs at least as extreme as observed.
    if observed >= 0:
        p_value = sum(1 for d in diffs if d <= 0) / n_resamples
    else:
        p_value = sum(1 for d in diffs if d >= 0) / n_resamples
    p_value = min(1.0, 2 * p_value)

    return {
        "mean_a": sum(scores_a) / n,
        "mean_b": sum(scores_b) / n,
        "mean_diff": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n": float(n),
    }


def compare_preset_scores(
    scores_by_preset: dict[str, list[dict[str, float | str]]],
    *,
    baseline: str,
    challenger: str,
    language: str | None = None,
    metric: MetricName = "hit",
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap: baseline vs challenger on one metric, optionally per language."""
    base_rows = scores_by_preset[baseline]
    chall_rows = scores_by_preset[challenger]

    if language is not None:
        base_rows = [r for r in base_rows if r["language"] == language]
        chall_rows = [r for r in chall_rows if r["language"] == language]

    base_scores = [float(r[metric]) for r in base_rows]
    chall_scores = [float(r[metric]) for r in chall_rows]
    result = bootstrap_mean_diff(
        base_scores,
        chall_scores,
        n_resamples=n_resamples,
        seed=seed,
    )
    result["metric"] = metric  # type: ignore[assignment]
    result["baseline"] = baseline  # type: ignore[assignment]
    result["challenger"] = challenger  # type: ignore[assignment]
    if language:
        result["language"] = language  # type: ignore[assignment]
    return result
