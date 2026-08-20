"""Shared comparison/reporting helpers for the eval harnesses.

Extracted from ``eval/compare.py`` so that the preset, chunking, and retriever
comparisons all share one implementation instead of reaching across modules for
private helpers.
"""

from __future__ import annotations

import sys

from eval.significance import compare_preset_scores


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def pairwise_significance(
    scores_by_variant: dict[str, list[dict]],
    variants: tuple[str, ...],
    *,
    baseline: str,
    languages: tuple[str, ...] = ("hi", "gu"),
    metrics: tuple[str, ...] = ("hit", "mrr"),
    n_resamples: int = 10_000,
) -> dict:
    """Paired bootstrap of the baseline against every other variant."""
    comparisons: dict[str, dict] = {}
    for challenger in variants:
        if challenger == baseline:
            continue
        key = f"{baseline}_vs_{challenger}"
        comparisons[key] = {}
        for metric in metrics:
            comparisons[key][metric] = {
                "overall": compare_preset_scores(
                    scores_by_variant,
                    baseline=baseline,
                    challenger=challenger,
                    metric=metric,  # type: ignore[arg-type]
                    n_resamples=n_resamples,
                ),
                "by_language": {
                    lang: compare_preset_scores(
                        scores_by_variant,
                        baseline=baseline,
                        challenger=challenger,
                        language=lang,
                        metric=metric,  # type: ignore[arg-type]
                        n_resamples=n_resamples,
                    )
                    for lang in languages
                },
            }
    return comparisons
