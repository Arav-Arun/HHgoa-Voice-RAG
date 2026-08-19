"""Passage-aware chunker for MS MARCO-XI style documents."""

from __future__ import annotations

from core.chunking.base import BaseChunker
from core.chunking.semantic import SemanticChunker
from core.types import Chunk, Document

_METADATA_KEYS = (
    "source",
    "split",
    "query_id",
    "passage_index",
    "is_selected",
    "query",
    "Eng_Query",
    "Eng_passage",
)


class MetadataAwareChunker(BaseChunker):
    """Keep passages atomic when possible; semantic sub-split only when over max size.

    Propagates passage/query metadata onto every chunk for traceability and
    future filtering — without injecting query text into the embedded passage body.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._fallback = SemanticChunker(chunk_size=chunk_size, overlap=overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        passage_meta = {
            key: document.metadata[key]
            for key in _METADATA_KEYS
            if key in document.metadata
        }

        if len(text) <= self.chunk_size:
            return [
                Chunk(
                    id=f"{document.id}__0",
                    text=text,
                    document_id=document.id,
                    language=document.language,
                    metadata={
                        **passage_meta,
                        "chunking": "metadata_atomic",
                        "passage_atomic": True,
                    },
                )
            ]

        chunks = self._fallback.chunk(document)
        for chunk in chunks:
            chunk.metadata = {
                **passage_meta,
                **chunk.metadata,
                "chunking": "metadata_semantic_fallback",
                "passage_atomic": False,
            }
        return chunks
