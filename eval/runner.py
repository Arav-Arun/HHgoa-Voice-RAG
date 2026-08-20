"""Eval runner."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_retriever
from eval.dataset import load_eval_set
from eval.metrics import aggregate_scores, aggregate_scores_by_language
from eval.significance import score_examples


def run_eval(
    eval_path: Path,
    *,
    settings: Settings | None = None,
    top_k: int = 5,
) -> dict:
    examples = load_eval_set(eval_path)
    settings = settings or get_settings()
    retriever = build_retriever(settings)

    def retrieve_fn(query, top_k=top_k, language=None):
        return retriever.retrieve(query, top_k=top_k, language=language)

    rows = score_examples(examples, retrieve_fn, top_k=top_k)
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_preset": settings.embedding_preset,
        "embedding_model": settings.embedding_model or None,
        "overall": aggregate_scores(rows, top_k=top_k),
        "by_language": aggregate_scores_by_language(rows, top_k=top_k),
    }
