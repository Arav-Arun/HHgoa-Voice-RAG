"""Eval runner."""

from __future__ import annotations

from pathlib import Path

from core.factory import build_retriever
from eval.dataset import load_eval_set
from eval.metrics import evaluate_examples


def run_eval(eval_path: Path, *, top_k: int = 5) -> dict[str, float]:
    examples = load_eval_set(eval_path)
    retriever = build_retriever()
    return evaluate_examples(
        examples,
        lambda query, top_k=top_k: retriever.retrieve(query, top_k=top_k),
        top_k=top_k,
    )
