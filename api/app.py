"""FastAPI app — swap routes or add auth here."""

from __future__ import annotations

from fastapi import FastAPI

from api.schemas import QueryRequest, QueryResponse, SourceChunk
from core.config import get_settings
from core.factory import build_rag_pipeline

app = FastAPI(title="hhgoa RAG", version="0.1.0")
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_rag_pipeline(use_template_llm=False)
    return _pipeline


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "languages": ",".join(settings.supported_languages),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    response = get_pipeline().query(
        request.question,
        language=request.language,
        top_k=request.top_k,
    )
    return QueryResponse(
        query=response.query,
        answer=response.answer,
        language=response.language,
        sources=[
            SourceChunk(
                id=s.chunk.id,
                text=s.chunk.text,
                document_id=s.chunk.document_id,
                score=s.score,
                language=s.chunk.language,
            )
            for s in response.sources
        ],
    )
