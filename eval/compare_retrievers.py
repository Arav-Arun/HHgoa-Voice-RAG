"""Compare dense / sparse / hybrid retrieval on the held-out eval set.

Unlike the chunking and embedding comparisons, this one needs **no re-ingest**:
all three retrievers read the same index, so every variant sees byte-identical
chunks and embeddings. The only thing that changes is how candidates are ranked,
which makes this an exactly-paired ablation and the bootstrap correspondingly
tight.
"""

from __future__ import annotations

import json
from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_retriever
from eval._report import log, pairwise_significance
from eval.dataset import load_eval_set
from eval.metrics import aggregate_scores, aggregate_scores_by_language
from eval.significance import score_examples

DEFAULT_OUTPUT_PATH = Path("data/eval/retriever-compare.json")


def compare_retrievers(
    retrievers: tuple[str, ...],
    eval_path: Path,
    *,
    settings: Settings | None = None,
    top_k: int = 5,
    baseline: str = "dense",
    bootstrap_resamples: int = 10_000,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
) -> dict:
    base = settings or get_settings()
    examples = load_eval_set(eval_path)
    results: dict[str, dict] = {}
    scores_by_retriever: dict[str, list[dict]] = {}

    for name in retrievers:
        log(f"[retriever-compare] {name}: scoring {len(examples)} held-out queries...")
        variant = base.model_copy(update={"retriever_provider": name})
        retriever = build_retriever(variant)
        # Bind `retriever` explicitly: the lambda is consumed within this
        # iteration, but a late-binding closure over a loop variable is a
        # trap waiting for the first person who defers the call.
        def retrieve_fn(query, top_k=top_k, language=None, _r=retriever):
            return _r.retrieve(query, top_k=top_k, language=language)

        per_example = score_examples(examples, retrieve_fn, top_k=top_k)
        scores_by_retriever[name] = per_example
        results[name] = {
            "retriever": name,
            "fusion": variant.fusion_method if name == "hybrid" else None,
            "candidate_k": variant.retrieval_candidate_k if name == "hybrid" else None,
            "overall": aggregate_scores(per_example, top_k=top_k),
            "by_language": aggregate_scores_by_language(per_example, top_k=top_k),
        }
        overall = results[name]["overall"]
        log(
            f"[retriever-compare] {name}: done "
            f"(hit@{top_k}={overall[f'hit@{top_k}']:.3f}, mrr={overall['mrr']:.3f})"
        )

    if baseline in retrievers and len(retrievers) > 1:
        log("[retriever-compare] running bootstrap significance tests...")
        results["significance"] = pairwise_significance(
            scores_by_retriever,
            retrievers,
            baseline=baseline,
            n_resamples=bootstrap_resamples,
        )

    results["embedding_preset"] = base.embedding_preset
    results["chunking_provider"] = base.chunking_provider

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return results
