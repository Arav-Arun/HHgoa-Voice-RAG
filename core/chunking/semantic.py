"""Sentence-boundary chunker with max-size packing and overlap."""

from __future__ import annotations

from core.chunking.base import BaseChunker
from core.chunking.sentences import split_sentences
from core.types import Chunk, Document


class SemanticChunker(BaseChunker):
    """Pack whole sentences into chunks up to chunk_size; overlap replays tail sentences."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = split_sentences(document.text)
        if not sentences:
            return []

        windows = self._pack_sentences(sentences)
        chunks: list[Chunk] = []
        for idx, (start, end, chunk_text) in enumerate(windows):
            chunks.append(
                Chunk(
                    id=f"{document.id}__{idx}",
                    text=chunk_text,
                    document_id=document.id,
                    language=document.language,
                    metadata={
                        **document.metadata,
                        "chunking": "semantic",
                        "sentence_start": start,
                        "sentence_end": end,
                    },
                )
            )
        return chunks

    def _pack_sentences(self, sentences: list[str]) -> list[tuple[int, int, str]]:
        if not sentences:
            return []

        windows: list[tuple[int, int, str]] = []
        start_idx = 0

        while start_idx < len(sentences):
            end_idx = start_idx
            chunk_parts: list[str] = []
            length = 0

            while end_idx < len(sentences):
                candidate = sentences[end_idx]
                extra = len(candidate) + (1 if chunk_parts else 0)
                if chunk_parts and length + extra > self.chunk_size:
                    break
                chunk_parts.append(candidate)
                length += extra
                end_idx += 1

            if not chunk_parts:
                chunk_parts = [sentences[start_idx]]
                end_idx = start_idx + 1

            windows.append((start_idx, end_idx, " ".join(chunk_parts)))

            if end_idx >= len(sentences):
                break

            next_start = end_idx
            if self.overlap > 0:
                overlap_len = 0
                overlap_start = end_idx
                while overlap_start > start_idx:
                    overlap_start -= 1
                    overlap_len += len(sentences[overlap_start]) + 1
                    if overlap_len >= self.overlap:
                        break
                next_start = max(start_idx + 1, overlap_start)

            start_idx = next_start

        return windows
