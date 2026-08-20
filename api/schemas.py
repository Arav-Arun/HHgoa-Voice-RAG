"""API request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    # Optional override. Left unset, the language is read off the script the
    # question is written in.
    language: str | None = Field(default=None, pattern="^(hi|gu)$")
    top_k: int | None = Field(default=None, ge=1, le=20)
    # fast: local extractive, budgeted for <200ms. quality: LLM tool-calling harness.
    mode: Literal["fast", "quality"] = "fast"


class SourceChunk(BaseModel):
    id: str
    text: str
    document_id: str
    score: float
    language: str
    # Per-strategy signals behind the fused score (dense cosine, BM25, ranks).
    components: dict[str, float] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    answer: str
    language: str
    sources: list[SourceChunk]
    abstained: bool = False
    guardrail_reason: str | None = None
    guardrail_stage: str | None = None
    citations: list[str] = Field(default_factory=list)
    # Observability: which path served this, and where the time went.
    path: str = "fast"
    trace_id: str = ""
    total_ms: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    quality: dict[str, Any] | None = None


class VoiceQueryResponse(QueryResponse):
    transcription: str
    stt_ms: float = 0.0
    end_to_end_ms: float = 0.0


class HealthResponse(BaseModel):
    status: str
    ready: bool
    languages: list[str]
    indexed_chunks: int
    bm25_terms: int
    retriever: str
    fusion: str
    chunking: str
    embedding_preset: str
    guardrail: str
    stt_provider: str
    quality_path_available: bool
