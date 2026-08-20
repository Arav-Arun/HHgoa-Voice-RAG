"""Compare chunking strategies on the held-out eval set.

Each strategy indexes into its own directory, so a run can be repeated against
the indexes it already built (``reuse_index``) instead of re-ingesting 100k+
chunks per strategy. ``ablate_dedupe`` scores every strategy twice, with
passage-level candidate collapsing on and off, which is what separates a
strategy's retrieval quality from the fan-out penalty described in
:mod:`core.retriever.base`.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_bm25_index, build_embedder, build_retriever, build_vector_store
from eval._report import log, pairwise_significance
from eval.dataset import load_eval_set
from eval.metrics import aggregate_scores, aggregate_scores_by_language
from eval.significance import bootstrap_mean_diff, score_examples
from eval.split import get_corpus_row_indices, load_split_config

DEFAULT_OUTPUT_PATH = Path("data/eval/chunk-compare.json")
STRATEGIES = ("fixed", "semantic", "metadata", "recursive", "parent_child", "token_window")


def _score(retriever, examples, top_k: int) -> list[dict]:
    def retrieve_fn(query, top_k=top_k, language=None):
        return retriever.retrieve(query, top_k=top_k, language=language)

    return score_examples(examples, retrieve_fn, top_k=top_k)


def _summarize(rows: list[dict], top_k: int) -> dict:
    return {
        "overall": aggregate_scores(rows, top_k=top_k),
        "by_language": aggregate_scores_by_language(rows, top_k=top_k),
    }


def compare_chunking_strategies(
    strategies: tuple[str, ...],
    eval_path: Path,
    *,
    settings: Settings | None = None,
    top_k: int = 5,
    baseline: str = "fixed",
    bootstrap_resamples: int = 10_000,
    index_root: str | Path | None = None,
    reuse_index: bool = False,
    ablate_dedupe: bool = False,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
) -> dict:
    """Ingest the corpus with each chunker and score the held-out queries."""
    from ingest.indexer import ingest_msmarco_xi

    base = settings or get_settings()
    split_config = load_split_config()
    # Each strategy builds into its own directory. Writing them all to
    # settings.index_dir would clobber the working index and leave whichever
    # strategy happened to run last in place.
    scratch_root = Path(index_root) if index_root else base.index_dir.parent / "index_chunkcmp"
    corpus_indices = get_corpus_row_indices(split_config)
    examples = load_eval_set(eval_path)
    # The encoder is the same for every strategy, so load the model once.
    embedder = build_embedder(base)

    results: dict[str, dict] = {}
    scores_by_strategy: dict[str, list[dict]] = {}

    for strategy in strategies:
        variant = base.model_copy(
            update={"chunking_provider": strategy, "index_dir": scratch_root / strategy}
        )
        if reuse_index and (variant.index_dir / "chunks.json").exists():
            log(f"[chunk-compare] {strategy}: reusing index at {variant.index_dir}")
        else:
            log(f"[chunk-compare] {strategy}: ingesting ({len(corpus_indices)} rows/lang)...")
            ingest_msmarco_xi(variant, split=split_config.split, row_indices=corpus_indices)

        # Share the loaded store and BM25 index across the dedupe ablation;
        # they are hundreds of megabytes and identical between the two runs.
        store = build_vector_store(variant)
        index = build_bm25_index(variant)
        log(f"[chunk-compare] {strategy}: scoring {len(examples)} held-out queries...")
        retriever = build_retriever(variant, store=store, embedder=embedder, index=index)
        per_example = _score(retriever, examples, top_k)
        scores_by_strategy[strategy] = per_example

        entry = {
            "chunking_provider": strategy,
            "chunks_indexed": store.count(),
            "fanout": round(retriever.fanout, 4),
            **_summarize(per_example, top_k),
        }

        if ablate_dedupe:
            log(f"[chunk-compare] {strategy}: re-scoring without passage collapsing...")
            raw = build_retriever(
                variant.model_copy(update={"retrieval_dedupe": False}),
                store=store,
                embedder=embedder,
                index=index,
            )
            raw_rows = _score(raw, examples, top_k)
            entry["without_dedupe"] = _summarize(raw_rows, top_k)
            entry["dedupe_delta"] = {
                metric: bootstrap_mean_diff(
                    [float(r[metric]) for r in per_example],
                    [float(r[metric]) for r in raw_rows],
                    n_resamples=bootstrap_resamples,
                )
                for metric in ("hit", "mrr")
            }

        results[strategy] = entry
        overall = entry["overall"]
        log(
            f"[chunk-compare] {strategy}: done "
            f"(hit@{top_k}={overall[f'hit@{top_k}']:.3f}, mrr={overall['mrr']:.3f})"
        )

    if baseline in strategies and len(strategies) > 1:
        log("[chunk-compare] running bootstrap significance tests...")
        results["significance"] = pairwise_significance(
            scores_by_strategy,
            strategies,
            baseline=baseline,
            n_resamples=bootstrap_resamples,
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return results
