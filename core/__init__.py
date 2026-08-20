"""Core RAG library, tweak submodules independently."""

from core.config import Settings, get_settings
from core.factory import (
    build_bm25_index,
    build_chat_clients,
    build_chunker,
    build_embedder,
    build_fast_answerer,
    build_guardrail,
    build_orchestrator,
    build_rag_pipeline,
    build_retriever,
    build_stt,
    build_vector_store,
)
from core.rag.pipeline import RAGPipeline
from core.types import Chunk, Document, RAGResponse, ScoredChunk

__all__ = [
    "Chunk",
    "Document",
    "RAGPipeline",
    "RAGResponse",
    "ScoredChunk",
    "Settings",
    "build_bm25_index",
    "build_chat_clients",
    "build_chunker",
    "build_embedder",
    "build_fast_answerer",
    "build_guardrail",
    "build_orchestrator",
    "build_rag_pipeline",
    "build_retriever",
    "build_stt",
    "build_vector_store",
    "get_settings",
]
