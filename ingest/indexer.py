"""Ingest pipeline: load → chunk → embed → index."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_chunker, build_embedder, build_vector_store
from core.types import Document
from core.vectorstore.memory import MemoryVectorStore
from ingest.loaders import (
    MSMARCO_XI_SCOPE_LANGUAGES,
    iter_msmarco_xi,
    load_directory,
    load_jsonl,
    load_text_file,
)


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


def ingest_msmarco_xi(
    settings: Settings | None = None,
    *,
    languages: tuple[str, ...] = MSMARCO_XI_SCOPE_LANGUAGES,
    split: str = "validation",
    limit: int | None = 100,
    batch_size: int = 256,
    save_index: bool = True,
) -> int:
    """Ingest Hindi/Gujarati passages from ai4bharat/MSMARCO-XI."""
    settings = settings or get_settings()
    chunker = build_chunker(settings)
    embedder = build_embedder(settings)
    store = MemoryVectorStore()
    total_chunks = 0

    for batch in iter_msmarco_xi(
        languages=languages,
        split=split,
        limit=limit,
        batch_size=batch_size,
    ):
        batch_chunks = []
        for document in batch:
            batch_chunks.extend(chunker.chunk(document))

        if not batch_chunks:
            continue

        embeddings = embedder.embed_texts([chunk.text for chunk in batch_chunks])
        store.add(batch_chunks, embeddings)
        total_chunks += len(batch_chunks)

    if save_index and total_chunks:
        settings.index_dir.mkdir(parents=True, exist_ok=True)
        store.save(str(settings.index_dir))

    return total_chunks
