"""Shared data types for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Raw document before chunking."""

    id: str
    text: str
    language: str = "hi"  # hi | gu, see docs/scope.md
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """Indexed text unit."""

    id: str
    text: str
    document_id: str
    language: str = "hi"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredChunk:
    """A chunk with its fused retrieval score.

    ``score`` is whatever the active retriever ranks by, for hybrid retrieval
    that is a fusion score, which is deliberately *not* on a calibrated scale.
    ``components`` carries the underlying per-strategy signals (dense cosine,
    BM25 score, and each one's rank) so downstream consumers can use the signal
    they actually need. The grounding guardrail in particular is calibrated
    against the dense cosine and must not be fed a rank-fusion score.
    """

    chunk: Chunk
    score: float
    components: dict[str, float] = field(default_factory=dict)

    @property
    def dense_score(self) -> float:
        """Calibrated dense cosine, falling back to ``score`` for dense-only retrieval."""
        return float(self.components.get("dense_score", self.score))


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[ScoredChunk]
    language: str = "hi"
    metadata: dict[str, Any] = field(default_factory=dict)
