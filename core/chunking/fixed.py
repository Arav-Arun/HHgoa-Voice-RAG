"""Fixed-size character chunker with overlap."""

from __future__ import annotations

from core.chunking.base import BaseChunker
from core.types import Chunk, Document


class FixedSizeChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        id=f"{document.id}__{idx}",
                        text=chunk_text,
                        document_id=document.id,
                        language=document.language,
                        metadata={**document.metadata, "char_start": start, "char_end": end},
                    )
                )
                idx += 1
            if end >= len(text):
                break
            start = max(end - self.overlap, start + 1)

        return chunks
