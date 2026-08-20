"""FastAPI service for the voice RAG pipeline.

Components are built once at startup and warmed with a dummy encode plus a dummy
search. Without that, the first real request pays the transformer load, roughly
nine seconds, and every latency number a caller sees is a lie about the second
request onward.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    VoiceQueryResponse,
)
from api.stats import collect_stats
from core.config import get_settings
from core.factory import (
    build_bm25_index,
    build_english_sources,
    build_rag_pipeline,
    build_stt,
)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _state["settings"] = settings
    _state["pipeline"] = build_rag_pipeline(settings)
    _state["stt"] = build_stt(settings)
    _state["bm25_terms"] = len(build_bm25_index(settings).vocab)
    # Display-only English source text; absent file just means none is shown.
    _state["english"] = build_english_sources(settings)

    # Warm the transformer and the index so request #1 is representative.
    warm_start = time.perf_counter()
    try:
        _state["pipeline"].warm(settings.default_language)
        _state["ready"] = True
    except Exception as exc:  # noqa: BLE001 - serve degraded rather than refuse to boot
        _state["ready"] = False
        _state["warmup_error"] = f"{type(exc).__name__}: {exc}"
    _state["warmup_ms"] = (time.perf_counter() - warm_start) * 1000.0

    yield
    _state.clear()


app = FastAPI(title="hhgoa voice RAG", version="0.2.0", lifespan=lifespan)

class _RevalidatingStatic(StaticFiles):
    """Static files that must be revalidated before reuse.

    StaticFiles already sends ETag and Last-Modified, but without an explicit
    Cache-Control browsers may serve a heuristically-cached copy and quietly
    ignore edits. "no-cache" still allows cheap 304s, it just forbids using a
    cached asset without asking first.
    """

    def is_not_modified(self, response_headers, request_headers) -> bool:
        response_headers.setdefault("cache-control", "no-cache")
        return super().is_not_modified(response_headers, request_headers)

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers.setdefault("cache-control", "no-cache")
        return response


# Static demo UI. Purely a client of the endpoints below; it holds no
# retrieval or model logic, and the API works identically without it.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", _RevalidatingStatic(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(
            _STATIC_DIR / "index.html", headers={"cache-control": "no-cache"}
        )


def get_pipeline():
    pipeline = _state.get("pipeline")
    if pipeline is None:  # pragma: no cover - only when lifespan did not run
        pipeline = _state["pipeline"] = build_rag_pipeline()
    return pipeline


def _english():
    english = _state.get("english")
    if english is None:
        english = _state["english"] = build_english_sources()
    return english


def get_stt():
    stt = _state.get("stt")
    if stt is None:  # pragma: no cover
        stt = _state["stt"] = build_stt()
    return stt


def _to_query_response(response) -> QueryResponse:
    meta = response.metadata
    guardrail = meta.get("guardrail", {})
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
                text_en=_english().get(s.chunk.document_id),
                components=s.components,
            )
            for s in response.sources
        ],
        abstained=bool(guardrail.get("blocked")),
        guardrail_reason=guardrail.get("reason"),
        guardrail_stage=guardrail.get("stage"),
        citations=meta.get("citations", []),
        path=meta.get("path", "fast"),
        trace_id=meta.get("trace_id", ""),
        total_ms=meta.get("total_ms", 0.0),
        timings_ms=meta.get("timings_ms", {}),
        quality=meta.get("quality"),
    )


def _read_audio(audio: UploadFile, data: bytes):
    if not data:
        raise HTTPException(status_code=400, detail="Audio payload is empty")
    return data, audio.content_type or "application/octet-stream", audio.filename or "audio.wav"


def _transcribe(data: bytes, audio: UploadFile, language: str | None):
    payload, content_type, filename = _read_audio(audio, data)
    start = time.perf_counter()
    try:
        result = get_stt().transcribe(
            payload, language=language, content_type=content_type, filename=filename
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result, (time.perf_counter() - start) * 1000.0


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = _state.get("settings") or get_settings()
    pipeline = _state.get("pipeline")
    chunks = 0
    if pipeline is not None:
        store = getattr(pipeline.retriever, "store", None)
        chunks = store.count() if store is not None else 0
    return HealthResponse(
        status="ok",
        ready=bool(_state.get("ready")),
        languages=list(settings.supported_languages),
        indexed_chunks=chunks,
        bm25_terms=_state.get("bm25_terms", 0),
        retriever=settings.retriever_provider,
        fusion=settings.fusion_method,
        chunking=settings.chunking_provider,
        embedding_preset=settings.embedding_preset,
        guardrail=settings.guardrail_provider,
        stt_provider=settings.stt_provider,
        quality_path_available=bool(getattr(get_pipeline().orchestrator, "chat_clients", [])),
    )


@app.get("/stats", include_in_schema=True)
def stats() -> dict:
    """Measured results, read from run artifacts. Never synthesised."""
    return collect_stats()


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    response = get_pipeline().query(
        request.question,
        language=request.language,
        top_k=request.top_k,
        mode=request.mode,
    )
    return _to_query_response(response)


@app.post("/voice-query", response_model=VoiceQueryResponse)
async def voice_query(
    language: str | None = Form(default=None, pattern="^(hi|gu)$"),
    top_k: int | None = Form(default=None),
    mode: str = Form(default="fast", pattern="^(fast|quality)$"),
    audio: UploadFile = File(...),  # noqa: B008 - FastAPI dependency idiom
) -> VoiceQueryResponse:
    started = time.perf_counter()
    transcription, stt_ms = _transcribe(await audio.read(), audio, language)

    if not transcription.text:
        raise HTTPException(status_code=400, detail="Transcription was empty")

    # The transcript's script decides, unless the caller forced a language.
    response = get_pipeline().query(
        transcription.text, language=language or transcription.language, top_k=top_k, mode=mode
    )
    base = _to_query_response(response)
    return VoiceQueryResponse(
        **base.model_dump(),
        transcription=transcription.text,
        stt_ms=round(stt_ms, 2),
        end_to_end_ms=round((time.perf_counter() - started) * 1000.0, 2),
    )
