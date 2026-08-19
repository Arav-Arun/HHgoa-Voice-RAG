"""API request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    language: str = Field(default="hi", pattern="^(hi|gu)$")
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    id: str
    text: str
    document_id: str
    score: float
    language: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    language: str
    sources: list[SourceChunk]
    abstained: bool = False
    guardrail_reason: str | None = None


class TranscribeResponse(BaseModel):
    text: str
    language: str
    provider: str


class VoiceQueryResponse(QueryResponse):
    transcription: str
