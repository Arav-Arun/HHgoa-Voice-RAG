"""Validate that eval labels are reachable in the built index."""

from __future__ import annotations

import json
from pathlib import Path

from core.config import get_settings
from eval.dataset import EvalExample, load_eval_set


def load_index_document_ids(index_dir: Path | None = None) -> set[str]:
    settings = get_settings()
    index_dir = index_dir or settings.index_dir
    chunks_path = index_dir / "chunks.json"
    if not chunks_path.exists():
        return set()
    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    return {str(row["document_id"]) for row in raw}


def validate_eval_coverage(
    examples: list[EvalExample],
    *,
    index_dir: Path | None = None,
) -> dict:
    doc_ids = load_index_document_ids(index_dir)
    missing: list[dict] = []
    covered = 0

    for example in examples:
        missing_ids = [doc_id for doc_id in example.expected_doc_ids if doc_id not in doc_ids]
        if missing_ids:
            missing.append(
                {
                    "query_id": example.query_id,
                    "language": example.language,
                    "missing_doc_ids": missing_ids,
                    "query": example.query[:80],
                }
            )
        else:
            covered += 1

    return {
        "queries": len(examples),
        "fully_covered": covered,
        "missing_labels": len(missing),
        "index_documents": len(doc_ids),
        "ok": len(missing) == 0,
        "missing": missing[:20],
    }


def validate_eval_file(
    eval_path: Path,
    *,
    index_dir: Path | None = None,
) -> dict:
    return validate_eval_coverage(load_eval_set(eval_path), index_dir=index_dir)
