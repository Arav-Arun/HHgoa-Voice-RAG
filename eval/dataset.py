"""Eval dataset format and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalExample:
    query: str
    expected_doc_ids: list[str]
    language: str = "hi"
    query_id: int | None = None
    split: str = "validation"

    def to_dict(self) -> dict:
        row = {
            "query": self.query,
            "language": self.language,
            "expected_doc_ids": self.expected_doc_ids,
        }
        if self.query_id is not None:
            row["query_id"] = self.query_id
        if self.split:
            row["split"] = self.split
        return row


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
                    query_id=row.get("query_id"),
                    split=str(row.get("split", "validation")),
                )
            )
    return examples
