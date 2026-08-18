"""Core RAG library — tweak submodules independently."""

from core.config import Settings, get_settings
from core.factory import (
    build_chunker,
    build_embedder,
    build_llm,
    build_rag_pipeline,
    build_retriever,
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
    "build_chunker",
    "build_embedder",
    "build_llm",
    "build_rag_pipeline",
    "build_retriever",
    "build_vector_store",
    "get_settings",
]
