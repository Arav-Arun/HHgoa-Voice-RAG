"""FastAPI app — swap routes or add auth here."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api.schemas import QueryRequest, QueryResponse, SourceChunk, TranscribeResponse, VoiceQueryResponse
from core.config import get_settings
from core.factory import build_rag_pipeline, build_stt

app = FastAPI(title="hhgoa RAG", version="0.1.0")
_pipeline = None
_stt = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_rag_pipeline(use_template_llm=False)
    return _pipeline


def get_stt():
    global _stt
    if _stt is None:
        _stt = build_stt()
    return _stt


def _to_query_response(response) -> QueryResponse:
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


@app.get("/health")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "ok",
        "languages": ",".join(settings.supported_languages),
        "stt_provider": settings.stt_provider,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    response = get_pipeline().query(
        request.question,
        language=request.language,
        top_k=request.top_k,
    )
    return _to_query_response(response)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    language: str = Form(default="hi", pattern="^(hi|gu)$"),
    audio: UploadFile = File(...),
) -> TranscribeResponse:
    audio_bytes = await audio.read()
    try:
        result = get_stt().transcribe(
            audio_bytes,
            language=language,
            content_type=audio.content_type or "application/octet-stream",
            filename=audio.filename or "audio.wav",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        provider=result.provider,
    )


@app.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(
    language: str = Form(default="hi", pattern="^(hi|gu)$"),
    top_k: int | None = Form(default=None),
    audio: UploadFile = File(...),
) -> VoiceQueryResponse:
    audio_bytes = await audio.read()
    try:
        transcription = get_stt().transcribe(
            audio_bytes,
            language=language,
            content_type=audio.content_type or "application/octet-stream",
            filename=audio.filename or "audio.wav",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not transcription.text:
        raise HTTPException(status_code=400, detail="Transcription was empty")

    response = get_pipeline().query(
        transcription.text,
        language=language,
        top_k=top_k,
    )
    base = _to_query_response(response)
    return VoiceQueryResponse(
        **base.model_dump(),
        transcription=transcription.text,
    )
