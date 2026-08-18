"""Build retrieval eval fixtures from MS MARCO-XI (is_selected = relevant)."""

from __future__ import annotations

import json
from pathlib import Path

from eval.dataset import EvalExample
from ingest.loaders import (
    MSMARCO_XI_SCOPE_LANGUAGES,
    load_msmarco_xi_rows,
    msmarco_passage_id,
)

DEFAULT_EVAL_PATH = Path("data/eval/queries.jsonl")


def eval_example_from_msmarco_row(row: dict, *, language: str) -> EvalExample | None:
    """Map one MS MARCO-XI example to an eval query using selected passages as labels."""
    query = str(row.get("query") or "").strip()
    if not query:
        return None

    passages = row.get("passages") or {}
    translated = passages.get("Translated_passages") or []
    is_selected = passages.get("is_selected") or []
    query_id = row["query_id"]

    expected_doc_ids: list[str] = []
    for idx, flag in enumerate(is_selected):
        if not flag:
            continue
        text = str(translated[idx]).strip() if idx < len(translated) else ""
        if not text:
            continue
        expected_doc_ids.append(msmarco_passage_id(language, query_id, idx))

    if not expected_doc_ids:
        return None

    return EvalExample(
        query=query,
        expected_doc_ids=expected_doc_ids,
        language=language,
    )


def build_msmarco_eval_set(
    *,
    languages: tuple[str, ...] = MSMARCO_XI_SCOPE_LANGUAGES,
    split: str = "validation",
    limit: int | None = 100,
) -> list[EvalExample]:
    """Build eval examples from the same MS MARCO-XI slice ingest indexes by default."""
    examples: list[EvalExample] = []
    for language in languages:
        for row in load_msmarco_xi_rows(language, split, limit=limit):
            example = eval_example_from_msmarco_row(row, language=language)
            if example is not None:
                examples.append(example)
    return examples


def write_eval_set(path: Path, examples: list[EvalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    {
                        "query": example.query,
                        "language": example.language,
                        "expected_doc_ids": example.expected_doc_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_and_write_msmarco_eval_set(
    path: Path = DEFAULT_EVAL_PATH,
    *,
    languages: tuple[str, ...] = MSMARCO_XI_SCOPE_LANGUAGES,
    split: str = "validation",
    limit: int | None = 100,
) -> list[EvalExample]:
    examples = build_msmarco_eval_set(languages=languages, split=split, limit=limit)
    write_eval_set(path, examples)
    return examples
