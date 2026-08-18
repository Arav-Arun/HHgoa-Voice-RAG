"""Shared data types for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Raw document before chunking."""

    id: str
    text: str
    language: str = "hi"  # hi | gu — see docs/scope.md
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
    chunk: Chunk
    score: float


@dataclass
class RAGResponse:
    query: str
    answer: str
    sources: list[ScoredChunk]
    language: str = "hi"
    metadata: dict[str, Any] = field(default_factory=dict)
