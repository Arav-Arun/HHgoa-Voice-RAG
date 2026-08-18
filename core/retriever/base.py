"""Retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import ScoredChunk


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        """Return ranked chunks for a query."""
