"""Embedding interface — swap providers without touching retrieval code."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]
