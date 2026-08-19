"""Compare chunking strategies on the held-out eval set."""

from __future__ import annotations

import sys
from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_retriever
from eval.compare import _log, _pairwise_significance
from eval.dataset import load_eval_set
from eval.metrics import aggregate_scores, aggregate_scores_by_language
from eval.significance import score_examples
from eval.split import get_corpus_row_indices, load_split_config


def compare_chunking_strategies(
    strategies: tuple[str, ...],
    eval_path: Path,
    *,
    settings: Settings | None = None,
    top_k: int = 5,
    baseline: str = "fixed",
    bootstrap_resamples: int = 10_000,
) -> dict:
    """Re-ingest corpus with each chunker and score held-out queries."""
    from ingest.indexer import ingest_msmarco_xi

    base = settings or get_settings()
    split_config = load_split_config()
    corpus_indices = get_corpus_row_indices(split_config)
    examples = load_eval_set(eval_path)
    results: dict[str, dict] = {}
    scores_by_strategy: dict[str, list[dict]] = {}

    for strategy in strategies:
        _log(f"[chunk-compare] strategy={strategy}: ingesting corpus ({len(corpus_indices)} rows/lang)...")
        strategy_settings = base.model_copy(update={"chunking_provider": strategy})
        ingest_msmarco_xi(
            strategy_settings,
            split=split_config.split,
            row_indices=corpus_indices,
        )
        _log(f"[chunk-compare] strategy={strategy}: scoring {len(examples)} held-out queries...")
        retriever = build_retriever(strategy_settings)
        retrieve_fn = lambda query, top_k=top_k: retriever.retrieve(query, top_k=top_k)
        per_example = score_examples(examples, retrieve_fn, top_k=top_k)
        scores_by_strategy[strategy] = per_example
        results[strategy] = {
            "chunking_provider": strategy,
            "overall": aggregate_scores(per_example, top_k=top_k),
            "by_language": aggregate_scores_by_language(per_example, top_k=top_k),
        }
        overall = results[strategy]["overall"]
        _log(
            f"[chunk-compare] strategy={strategy}: done "
            f"(hit@{top_k}={overall[f'hit@{top_k}']:.3f}, mrr={overall['mrr']:.3f})"
        )

    if baseline in strategies and len(strategies) > 1:
        _log("[chunk-compare] running bootstrap significance tests...")
        results["significance"] = _pairwise_significance(
            scores_by_strategy,
            strategies,
            baseline=baseline,
            n_resamples=bootstrap_resamples,
        )

    return results
