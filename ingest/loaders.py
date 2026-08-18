"""Document loaders — MS MARCO-XI and custom formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from core.types import Document

MSMARCO_XI_DATASET = "ai4bharat/MSMARCO-XI"
MSMARCO_XI_SCOPE_LANGUAGES = ("hi", "gu")

# Parquet paths inside the Hugging Face repo (see dataset card).
_MSMARCO_XI_FILES: dict[str, dict[str, str]] = {
    "hi": {
        "train": "train/hintrain.parquet",
        "validation": "validation/hinval.parquet",
    },
    "gu": {
        "train": "train/gujtrain.parquet",
        "validation": "validation/gujval.parquet",
    },
}


def msmarco_passage_id(language: str, query_id: object, passage_index: int) -> str:
    """Stable document ID shared by ingest and eval fixtures."""
    return f"{language}_{query_id}_p{passage_index}"


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


def _msmarco_example_to_documents(row: dict, *, language: str, split: str) -> list[Document]:
    """Expand one MS MARCO-XI example into passage-level documents."""
    query_id = row["query_id"]
    passages = row["passages"]
    translated = passages["Translated_passages"]
    is_selected = passages.get("is_selected", [])
    english_passages = passages.get("English_passages", [])

    documents: list[Document] = []
    for idx, text in enumerate(translated):
        passage_text = str(text).strip()
        if not passage_text:
            continue
        documents.append(
            Document(
                id=msmarco_passage_id(language, query_id, idx),
                text=passage_text,
                language=language,
                metadata={
                    "source": MSMARCO_XI_DATASET,
                    "split": split,
                    "query_id": query_id,
                    "passage_index": idx,
                    "is_selected": is_selected[idx] if idx < len(is_selected) else 0,
                    "query": row.get("query", ""),
                    "Eng_Query": row.get("Eng_Query", ""),
                    "Eng_passage": english_passages[idx] if idx < len(english_passages) else "",
                },
            )
        )
    return documents


def load_msmarco_xi_rows(language: str, split: str, *, limit: int | None) -> list[dict]:
    from datasets import load_dataset

    if language not in _MSMARCO_XI_FILES:
        supported = ", ".join(sorted(_MSMARCO_XI_FILES))
        raise ValueError(f"Unsupported MS MARCO-XI language {language!r}. Supported: {supported}")

    data_file = _MSMARCO_XI_FILES[language][split]
    split_arg = split if limit is None else f"{split}[:{limit}]"
    dataset = load_dataset(
        MSMARCO_XI_DATASET,
        data_files={split: data_file},
        split=split_arg,
    )
    return list(dataset)


def load_msmarco_xi(
    *,
    languages: tuple[str, ...] = MSMARCO_XI_SCOPE_LANGUAGES,
    split: str = "validation",
    limit: int | None = 100,
) -> list[Document]:
    """Load Hindi/Gujarati passages from ai4bharat/MSMARCO-XI.

    ``limit`` caps the number of query examples per language (each yields ~10 passages).
    Pass ``limit=None`` to load the full split.
    """
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    documents: list[Document] = []
    for language in languages:
        for row in load_msmarco_xi_rows(language, split, limit=limit):
            documents.extend(_msmarco_example_to_documents(row, language=language, split=split))
    return documents


def iter_msmarco_xi(
    *,
    languages: tuple[str, ...] = MSMARCO_XI_SCOPE_LANGUAGES,
    split: str = "validation",
    limit: int | None = 100,
    batch_size: int = 256,
) -> Iterator[list[Document]]:
    """Yield MS MARCO-XI documents in batches (for large full-split ingests)."""
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    for language in languages:
        rows = load_msmarco_xi_rows(language, split, limit=limit)
        batch: list[Document] = []
        for row in rows:
            batch.extend(_msmarco_example_to_documents(row, language=language, split=split))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
