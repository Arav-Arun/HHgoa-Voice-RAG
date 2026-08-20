"""Sweep the hybrid dense/lexical balance on the **dev** slice.

The optimum is per language, not global: e5-small is strong on Hindi and weak on
Gujarati, so a weight that helps one hurts the other. Reporting a single global
number would hide two significant and opposite effects.

This runs on dev only. The weights it picks are applied to the held-out eval
slice exactly once, when results are reported, so no hyperparameter here is ever
chosen by looking at eval.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_bm25_index, build_embedder, build_retriever, build_vector_store
from eval._report import log
from eval.dataset import load_eval_set
from eval.metrics import aggregate_scores, aggregate_scores_by_language
from eval.significance import score_examples

DEFAULT_OUTPUT_PATH = Path("data/eval/fusion-sweep.json")
DEFAULT_DEV_PATH = Path("data/eval/dev_queries.jsonl")
# 0.0 is BM25 only, 1.0 is dense only; the interesting structure is in between.
DEFAULT_GRID = tuple(round(0.05 * step, 2) for step in range(21))


def sweep_fusion_weight(
    dev_path: Path = DEFAULT_DEV_PATH,
    *,
    settings: Settings | None = None,
    top_k: int = 5,
    grid: tuple[float, ...] = DEFAULT_GRID,
    languages: tuple[str, ...] = ("hi", "gu"),
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
) -> dict:
    base = settings or get_settings()
    examples = load_eval_set(dev_path)
    # One index load and one model load for the whole sweep.
    store = build_vector_store(base)
    embedder = build_embedder(base)
    index = build_bm25_index(base)

    sweep: list[dict[str, float]] = []
    for weight in grid:
        variant = base.model_copy(
            update={
                "retriever_provider": "hybrid",
                "fusion_dense_weight": weight,
                "fusion_dense_weight_hi": weight,
                "fusion_dense_weight_gu": weight,
            }
        )
        retriever = build_retriever(variant, store=store, embedder=embedder, index=index)

        def retrieve_fn(query, top_k=top_k, language=None, _r=retriever):
            return _r.retrieve(query, top_k=top_k, language=language)

        rows = score_examples(examples, retrieve_fn, top_k=top_k)
        by_language = aggregate_scores_by_language(rows, top_k=top_k)
        entry: dict[str, float] = {"dense_weight": weight}
        for language in languages:
            stats = by_language.get(language, {})
            entry[f"{language}_hit@{top_k}"] = round(stats.get(f"hit@{top_k}", 0.0), 4)
            entry[f"{language}_mrr"] = round(stats.get("mrr", 0.0), 4)
        entry[f"overall_hit@{top_k}"] = round(aggregate_scores(rows, top_k=top_k)[f"hit@{top_k}"], 4)
        sweep.append(entry)
        log(f"[fusion-sweep] w={weight:.2f} overall_hit@{top_k}={entry[f'overall_hit@{top_k}']:.4f}")

    # Ties go to the *lower* dense weight: keeping the lexical half alive is the
    # safer default on a corpus where the encoder is uneven across languages.
    best = {
        language: min(
            (row for row in sweep),
            key=lambda row, lg=language: (-row[f"{lg}_hit@{top_k}"], row["dense_weight"]),
        )["dense_weight"]
        for language in languages
    }

    result = {
        "slice": "dev",
        "queries": len(examples),
        "rrf_k": base.fusion_rrf_k,
        "candidate_k": base.retrieval_candidate_k,
        "chunking_provider": base.chunking_provider,
        "best_dense_weight": best,
        "sweep": sweep,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
