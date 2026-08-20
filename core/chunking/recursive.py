"""Recursive character chunker with a separator cascade.

Splits on the most semantically meaningful boundary that fits, and only falls
through to a cruder one when a piece is still too large:

    paragraph -> danda / sentence -> clause (comma, semicolon) -> whitespace -> raw characters

This is the standard strong baseline, and it degrades gracefully: a passage with
no punctuation at all still gets chunked, just at the character level. The
separator list is Indic-aware, the danda is the sentence terminator in
Devanagari and Gujarati text, and a Latin-only cascade would miss every sentence
boundary in this corpus.
"""

from __future__ import annotations

from itertools import pairwise

from core.chunking.base import BaseChunker
from core.types import Chunk, Document

DEFAULT_SEPARATORS = ("\n\n", "\n", "। ", "॥ ", "। ", ". ", "? ", "! ", "; ", ", ", " ", "")


class RecursiveChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: tuple[str, ...] = DEFAULT_SEPARATORS,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators

    def _split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        if not separators:
            # Out of separators: hard-cut at the size limit.
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        parts = text.split(separator)
        if len(parts) == 1:
            return self._split(text, rest)

        pieces: list[str] = []
        buffer = ""
        for part in parts:
            candidate = f"{buffer}{separator}{part}" if buffer else part
            if len(candidate) <= self.chunk_size:
                buffer = candidate
                continue
            if buffer:
                pieces.append(buffer)
            # The single part may still exceed the limit, recurse on it.
            if len(part) > self.chunk_size:
                pieces.extend(self._split(part, rest))
                buffer = ""
            else:
                buffer = part
        if buffer:
            pieces.append(buffer)
        return [p for p in pieces if p.strip()]

    def _apply_overlap(self, pieces: list[str]) -> list[str]:
        if self.overlap <= 0 or len(pieces) < 2:
            return pieces
        merged = [pieces[0]]
        for previous, current in pairwise(pieces):
            tail = previous[-self.overlap :]
            merged.append(f"{tail} {current}".strip())
        return merged

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        pieces = self._apply_overlap(self._split(text, self.separators))
        return [
            Chunk(
                id=f"{document.id}__{idx}",
                text=piece,
                document_id=document.id,
                language=document.language,
                metadata={**document.metadata, "chunking": "recursive", "piece_index": idx},
            )
            for idx, piece in enumerate(pieces)
        ]
