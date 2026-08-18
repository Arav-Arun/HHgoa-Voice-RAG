"""Ingest pipeline: load → chunk → embed → index."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_chunker, build_embedder, build_vector_store
from core.types import Document
from ingest.loaders import load_directory, load_jsonl, load_text_file


def ingest_documents(
    documents: list[Document],
    settings: Settings | None = None,
    *,
    save_index: bool = True,
) -> int:
    """Index documents and optionally persist to disk. Returns chunk count."""
    settings = settings or get_settings()
    chunker = build_chunker(settings)
    embedder = build_embedder(settings)
    store = build_vector_store(settings)

    all_chunks = []
    for document in documents:
        all_chunks.extend(chunker.chunk(document))

    if not all_chunks:
        return 0

    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_texts(texts)
    store.add(all_chunks, embeddings)

    if save_index:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        store.save(str(settings.index_dir))

    return len(all_chunks)


def ingest_path(
    source: Path,
    settings: Settings | None = None,
    *,
    language: str | None = None,
) -> int:
    """Ingest a file or directory."""
    settings = settings or get_settings()
    lang = language or settings.default_language

    if source.is_dir():
        documents = load_directory(source, default_language=lang)
    elif source.suffix == ".jsonl":
        documents = load_jsonl(source, default_language=lang)
    else:
        documents = [load_text_file(source, language=lang)]

    return ingest_documents(documents, settings)
