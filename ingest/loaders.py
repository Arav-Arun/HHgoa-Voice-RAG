"""Document loaders — add MS MARCO-XI / custom formats here."""

from __future__ import annotations

import json
from pathlib import Path

from core.types import Document


def load_text_file(path: Path, *, language: str = "hi", doc_id: str | None = None) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    return Document(
        id=doc_id or path.stem,
        text=text,
        language=language,
        metadata={"source": str(path), "format": "text"},
    )


def load_jsonl(path: Path, *, default_language: str = "hi") -> list[Document]:
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            documents.append(
                Document(
                    id=str(row.get("id", f"{path.stem}_{line_no}")),
                    text=str(row["text"]),
                    language=str(row.get("language", default_language)),
                    metadata={k: v for k, v in row.items() if k not in {"id", "text", "language"}},
                )
            )
    return documents


def load_directory(
    directory: Path,
    *,
    default_language: str = "hi",
    extensions: tuple[str, ...] = (".txt", ".jsonl"),
) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".txt" and ".txt" in extensions:
            documents.append(load_text_file(path, language=default_language))
        elif path.suffix == ".jsonl" and ".jsonl" in extensions:
            documents.extend(load_jsonl(path, default_language=default_language))
    return documents
