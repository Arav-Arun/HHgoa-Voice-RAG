"""Chunking interface, implement BaseChunker to plug in new strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Chunk, Document


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into retrievable chunks."""
