"""Compare local embedding presets on the same eval slice."""

from __future__ import annotations

import sys
from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_retriever
from eval.dataset import load_eval_set
from eval.metrics import aggregate_scores, aggregate_scores_by_language
from eval.significance import compare_preset_scores, score_examples


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pairwise_significance(
    scores_by_preset: dict[str, list[dict]],
    presets: tuple[str, ...],
    *,
    baseline: str,
    languages: tuple[str, ...] = ("hi", "gu"),
    metrics: tuple[str, ...] = ("hit", "mrr"),
    n_resamples: int = 10_000,
) -> dict:
    """Bootstrap baseline vs each other preset, per language and overall."""
    comparisons: dict[str, dict] = {}
    for challenger in presets:
        if challenger == baseline:
            continue
        key = f"{baseline}_vs_{challenger}"
        comparisons[key] = {}
        for metric in metrics:
            comparisons[key][metric] = {
                "overall": compare_preset_scores(
                    scores_by_preset,
                    baseline=baseline,
                    challenger=challenger,
                    metric=metric,  # type: ignore[arg-type]
                    n_resamples=n_resamples,
                ),
                "by_language": {
                    lang: compare_preset_scores(
                        scores_by_preset,
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


def compare_embedding_presets(
    presets: tuple[str, ...],
    eval_path: Path,
    *,
    settings: Settings | None = None,
    top_k: int = 5,
    ingest_limit: int = 500,
    ingest_split: str = "validation",
    baseline: str = "e5-small",
    bootstrap_resamples: int = 10_000,
) -> dict:
    """Re-ingest and eval each preset; include paired bootstrap vs baseline."""
    from ingest.indexer import ingest_msmarco_xi

    base = settings or get_settings()
    examples = load_eval_set(eval_path)
    results: dict[str, dict] = {}
    scores_by_preset: dict[str, list[dict]] = {}

    for preset in presets:
        _log(f"[eval-compare] preset={preset}: ingesting (limit={ingest_limit}/lang)...")
        preset_settings = base.model_copy(
            update={
                "embedding_provider": "sentence_transformers",
                "embedding_preset": preset,
                "embedding_model": "",
            }
        )
        ingest_msmarco_xi(
            preset_settings,
            split=ingest_split,
            limit=ingest_limit,
        )
        _log(f"[eval-compare] preset={preset}: scoring {len(examples)} queries...")
        retriever = build_retriever(preset_settings)
        retrieve_fn = lambda query, top_k=top_k: retriever.retrieve(query, top_k=top_k)
        per_example = score_examples(examples, retrieve_fn, top_k=top_k)
        scores_by_preset[preset] = per_example
        results[preset] = {
            "embedding_provider": preset_settings.embedding_provider,
            "embedding_preset": preset,
            "overall": aggregate_scores(per_example, top_k=top_k),
            "by_language": aggregate_scores_by_language(per_example, top_k=top_k),
        }
        overall = results[preset]["overall"]
        _log(
            f"[eval-compare] preset={preset}: done "
            f"(hit@{top_k}={overall[f'hit@{top_k}']:.3f}, mrr={overall['mrr']:.3f})"
        )

    if baseline in presets and len(presets) > 1:
        _log("[eval-compare] running bootstrap significance tests...")
        results["significance"] = _pairwise_significance(
            scores_by_preset,
            presets,
            baseline=baseline,
            n_resamples=bootstrap_resamples,
        )

    return results
