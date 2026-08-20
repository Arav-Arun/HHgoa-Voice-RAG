"""Sweep the ``parent_child`` child window size, reported per language.

Measured fairly, ``parent_child`` is the best strategy on Hindi and the worst on
Gujarati. The hypothesis is that this is a window-size effect rather than a flaw
in small-to-big retrieval: e5-small is weak on Gujarati, and a two-sentence
child may carry less signal than that encoder can place, while the same child is
plenty for a language it handles well.

Method mirrors the fusion-weight sweep, and for the same reason. **The window
size for each language is chosen on the dev slice**, then the chosen
configuration is scored once on the held-out eval slice. Selecting a
hyperparameter on eval and reporting it there would turn a held-out score into a
training score, so every size is scored on both slices and only dev is allowed
to decide.

``child_stride`` is held at 1 throughout, so window size is the only variable.

The sweep scores one uniform window size at a time and lets dev pick the best
per language, which is how the fusion weight is chosen too. It does not build a
mixed-window index: ``parent_child`` is not the shipping chunker, so the
question here is whether the window size *should* differ by language, not what a
production mixed index would score.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_bm25_index, build_embedder, build_retriever, build_vector_store
from eval._report import log
from eval.dataset import load_eval_set
from eval.metrics import aggregate_scores, aggregate_scores_by_language
from eval.significance import bootstrap_mean_diff, score_examples
from eval.split import get_corpus_row_indices, load_split_config

DEFAULT_OUTPUT_PATH = Path("data/eval/child-size-sweep.json")
DEFAULT_INDEX_ROOT = Path("data/index_childsweep")
DEFAULT_DEV_PATH = Path("data/eval/dev_queries.jsonl")
DEFAULT_REFERENCE_INDEX = Path("data/index_chunkcmp/fixed")
DEFAULT_SIZES = (2, 3)


def _summarize(rows: list[dict], top_k: int) -> dict:
    return {
        "overall": aggregate_scores(rows, top_k=top_k),
        "by_language": aggregate_scores_by_language(rows, top_k=top_k),
    }


def _hits(rows: list[dict], language: str | None = None) -> list[float]:
    picked = rows if language is None else [r for r in rows if r["language"] == language]
    return [float(r["hit"]) for r in picked]


def _paired_deltas(challenger, baseline, *, languages, n_resamples) -> dict:
    """Bootstrap challenger minus baseline on hit@k, overall and per language."""
    return {
        "overall": bootstrap_mean_diff(_hits(challenger), _hits(baseline), n_resamples=n_resamples),
        "by_language": {
            lang: bootstrap_mean_diff(
                _hits(challenger, lang), _hits(baseline, lang), n_resamples=n_resamples
            )
            for lang in languages
        },
    }


def sweep_child_size(
    eval_path: Path,
    dev_path: Path = DEFAULT_DEV_PATH,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    settings: Settings | None = None,
    top_k: int = 5,
    index_root: Path = DEFAULT_INDEX_ROOT,
    reference_index: Path | None = DEFAULT_REFERENCE_INDEX,
    reference_label: str = "fixed",
    baseline_size: int = 2,
    reuse_index: bool = True,
    languages: tuple[str, ...] = ("hi", "gu"),
    bootstrap_resamples: int = 10_000,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
) -> dict:
    from ingest.indexer import ingest_msmarco_xi

    base = settings or get_settings()
    split_config = load_split_config()
    corpus_indices = get_corpus_row_indices(split_config)
    dev_examples = load_eval_set(dev_path)
    eval_examples = load_eval_set(eval_path)
    embedder = build_embedder(base)

    variants: dict[str, dict] = {}
    eval_scores: dict[str, list[dict]] = {}
    dev_scores: dict[str, list[dict]] = {}

    def build_index(label: str, variant: Settings) -> Settings:
        variant = variant.model_copy(update={"index_dir": index_root / label})
        if reuse_index and (variant.index_dir / "chunks.json").exists():
            log(f"[child-sweep] {label}: reusing index at {variant.index_dir}")
        else:
            log(f"[child-sweep] {label}: ingesting ({len(corpus_indices)} rows/lang)...")
            ingest_msmarco_xi(variant, split=split_config.split, row_indices=corpus_indices)
        return variant

    def score(label: str, variant: Settings, window: dict[str, int]) -> None:
        store = build_vector_store(variant)
        index = build_bm25_index(variant)
        retriever = build_retriever(variant, store=store, embedder=embedder, index=index)

        def retrieve_fn(query, top_k=top_k, language=None):
            return retriever.retrieve(query, top_k=top_k, language=language)

        dev_rows = score_examples(dev_examples, retrieve_fn, top_k=top_k)
        eval_rows = score_examples(eval_examples, retrieve_fn, top_k=top_k)
        dev_scores[label] = dev_rows
        eval_scores[label] = eval_rows
        variants[label] = {
            "child_sentences": window,
            "chunks_indexed": store.count(),
            "fanout": round(retriever.fanout, 4),
            "dev": _summarize(dev_rows, top_k),
            "eval": _summarize(eval_rows, top_k),
        }
        dev_lang = variants[label]["dev"]["by_language"]
        log(
            f"[child-sweep] {label}: dev hit@{top_k}="
            f"{variants[label]['dev']['overall'][f'hit@{top_k}']:.4f} "
            + " ".join(f"{lg}={dev_lang.get(lg, {}).get(f'hit@{top_k}', 0.0):.4f}" for lg in languages)
        )

    for size in sizes:
        label = f"cs{size}"
        variant = build_index(
            label,
            base.model_copy(
                update={
                    "chunking_provider": "parent_child",
                    "child_sentences": size,
                    "child_stride": 1,
                    # A uniform sweep point: both languages get the same window.
                    "child_sentences_hi": size,
                    "child_sentences_gu": size,
                }
            ),
        )
        score(label, variant, dict.fromkeys(languages, size))

    # Selection happens on dev, and only on dev.
    best_by_language = {
        lang: max(
            sizes,
            key=lambda size, lg=lang: variants[f"cs{size}"]["dev"]["by_language"]
            .get(lg, {})
            .get(f"hit@{top_k}", 0.0),
        )
        for lang in languages
    }
    log(f"[child-sweep] dev-selected windows: {best_by_language}")

    if reference_index is not None and (reference_index / "chunks.json").exists():
        log(f"[child-sweep] {reference_label}: scoring reference index {reference_index}")
        score(
            reference_label,
            base.model_copy(update={"chunking_provider": "fixed", "index_dir": reference_index}),
            {},
        )

    # Everything reported below is the held-out eval slice.
    baseline_label = f"cs{baseline_size}"
    comparisons = {
        f"{label}_vs_{baseline_label}": _paired_deltas(
            eval_scores[label],
            eval_scores[baseline_label],
            languages=languages,
            n_resamples=bootstrap_resamples,
        )
        for label in variants
        if label != baseline_label
    }
    if reference_label in eval_scores:
        for label in variants:
            if label == reference_label:
                continue
            comparisons[f"{label}_vs_{reference_label}"] = _paired_deltas(
                eval_scores[label],
                eval_scores[reference_label],
                languages=languages,
                n_resamples=bootstrap_resamples,
            )

    payload = {
        "metric": "hit",
        "top_k": top_k,
        "child_stride": 1,
        "dev_queries": len(dev_examples),
        "eval_queries": len(eval_examples),
        "selected_on": "dev",
        "dev_selected_child_sentences": best_by_language,
        "baseline": baseline_label,
        "variants": variants,
        "comparisons": comparisons,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
