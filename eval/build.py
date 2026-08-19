"""Build held-out retrieval eval fixtures from MS MARCO-XI."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from eval.dataset import EvalExample
from eval.split import (
    DEFAULT_SPLIT,
    DEFAULT_SPLIT_PATH,
    EvalSplitConfig,
    get_eval_row_indices,
    get_validation_rows,
    write_split_config,
)
from ingest.loaders import load_msmarco_xi_rows_by_indices, msmarco_passage_id

DEFAULT_EVAL_PATH = Path("data/eval/queries.jsonl")


def eval_example_from_msmarco_row(
    row: dict,
    *,
    language: str,
    split: str,
) -> EvalExample | None:
    """Map one MS MARCO-XI example to query → relevant passage labels."""
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
        query_id=int(query_id),
        split=split,
    )


def build_held_out_eval_set(
    config: EvalSplitConfig = DEFAULT_SPLIT,
) -> list[EvalExample]:
    """Build eval queries from the held-out shuffled slice (disjoint from dev)."""
    eval_indices = get_eval_row_indices(config)
    examples: list[EvalExample] = []
    for language in config.languages:
        for row in load_msmarco_xi_rows_by_indices(language, config.split, eval_indices):
            example = eval_example_from_msmarco_row(
                row,
                language=language,
                split=config.split,
            )
            if example is not None:
                examples.append(example)
    return examples


def write_eval_set(path: Path, examples: list[EvalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")


def build_and_write_held_out_eval(
    path: Path = DEFAULT_EVAL_PATH,
    *,
    split_path: Path = DEFAULT_SPLIT_PATH,
    config: EvalSplitConfig | None = None,
) -> list[EvalExample]:
    """Write split.json + held-out queries.jsonl."""
    config = config or DEFAULT_SPLIT
    validation_rows = get_validation_rows(config)
    config = replace(config, validation_rows=validation_rows)
    write_split_config(split_path, config)
    examples = build_held_out_eval_set(config)
    write_eval_set(path, examples)
    return examples
