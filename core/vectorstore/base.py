"""Vector store interface — add FAISS/Qdrant/etc. by subclassing."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Chunk, ScoredChunk


class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks with precomputed embeddings."""

    @abstractmethod
    def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]:
        """Return top-k chunks by similarity."""

    @abstractmethod
    def save(self, path: str | None = None) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: str | None = None) -> None:
        """Load index from disk."""

    @abstractmethod
    def count(self) -> int:
        """Number of indexed chunks."""
