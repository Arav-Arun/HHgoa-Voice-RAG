"""Eval runner."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_retriever
from eval.dataset import load_eval_set
from eval.metrics import evaluate_examples, evaluate_examples_by_language


def run_eval(
    eval_path: Path,
    *,
    settings: Settings | None = None,
    top_k: int = 5,
) -> dict:
    examples = load_eval_set(eval_path)
    settings = settings or get_settings()
    retriever = build_retriever(settings)
    retrieve_fn = lambda query, top_k=top_k: retriever.retrieve(query, top_k=top_k)
    return {
        "embedding_provider": settings.embedding_provider,
        "embedding_preset": settings.embedding_preset,
        "embedding_model": settings.embedding_model or None,
        "overall": evaluate_examples(examples, retrieve_fn, top_k=top_k),
        "by_language": evaluate_examples_by_language(examples, retrieve_fn, top_k=top_k),
    }
