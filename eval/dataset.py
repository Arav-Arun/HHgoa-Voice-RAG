"""Eval dataset format and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalExample:
    query: str
    expected_doc_ids: list[str]
    language: str = "hi"


def load_eval_set(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(
                EvalExample(
                    query=row["query"],
                    expected_doc_ids=list(row.get("expected_doc_ids", [])),
                    language=str(row.get("language", "hi")),
                )
            )
    return examples
