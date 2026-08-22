"""Factory, swap components by editing provider strings in .env or here."""

from __future__ import annotations

import sys

from core.chunking.fixed import FixedSizeChunker
from core.chunking.metadata import MetadataAwareChunker
from core.chunking.parent_child import ParentChildChunker
from core.chunking.presets import CHUNKING_PRESETS
from core.chunking.recursive import RecursiveChunker
from core.chunking.semantic import SemanticChunker
from core.chunking.token_window import TokenWindowChunker
from core.config import Settings, get_settings
from core.embeddings.hash import HashEmbedder
from core.embeddings.onnx import OnnxEmbedder
from core.embeddings.openai import OpenAIEmbedder
from core.embeddings.presets import EMBEDDING_PRESETS
from core.embeddings.sentence_transformers import SentenceTransformerEmbedder
from core.english import FILENAME as _EN_FILE
from core.english import EnglishSources
from core.guardrails.composite import CompositeGuardrail
from core.guardrails.confidence import ConfidenceGate
from core.guardrails.cross_encoder import CrossEncoderScorer
from core.guardrails.grounding import GroundingGate
from core.guardrails.hallucination import HallucinationChecker
from core.guardrails.input_intent import InputIntentFilter
from core.guardrails.stub import StubGuardrail
from core.harness.cache import AnswerCache
from core.harness.orchestrator import Orchestrator
from core.llm.chat import ChatClient
from core.llm.extractive import ExtractiveAnswerer
from core.rag.pipeline import RAGPipeline
from core.retriever.base import BaseRetriever
from core.retriever.dense import DenseRetriever
from core.retriever.hybrid import HybridRetriever
from core.retriever.sparse import BM25Index, SparseRetriever
from core.stt.elevenlabs import ElevenLabsSTT
from core.stt.openai_whisper import OpenAIWhisperSTT
from core.stt.stub import StubSTT
from core.vectorstore.memory import MemoryVectorStore


def build_embedder(settings: Settings | None = None):
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()

    if provider in {"sentence_transformers", "local", "st"}:
        preset_name = settings.embedding_preset.lower()
        preset = EMBEDDING_PRESETS.get(preset_name)
        model_name = settings.embedding_model or (preset.model if preset else "")
        if not model_name:
            raise ValueError(
                "Set EMBEDDING_MODEL or a known EMBEDDING_PRESET "
                f"({', '.join(sorted(EMBEDDING_PRESETS))})"
            )
        shared = {
            "model_name": model_name,
            "query_prefix": preset.query_prefix if preset else "",
            "passage_prefix": preset.passage_prefix if preset else "",
            "batch_size": settings.embedding_batch_size,
            "device": settings.torch_device,
        }
        if settings.embedding_runtime.lower() in {"auto", "onnx"}:
            # Falls back to torch on its own when no export is present, so this
            # is safe to leave on by default.
            return OnnxEmbedder(threads=settings.embedding_threads, **shared)
        return SentenceTransformerEmbedder(**shared)
    if provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.llm_api_key,
            model=settings.embedding_model,
            base_url=settings.llm_base_url,
        )
    if provider == "hash":
        return HashEmbedder(dimension=settings.embedding_dim)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
        "Supported: sentence_transformers, hash, openai"
    )


def build_vector_store(settings: Settings | None = None):
    settings = settings or get_settings()
    provider = settings.vector_store.lower()

    if provider == "memory":
        store = MemoryVectorStore()
        if settings.index_dir.exists():
            store.load(str(settings.index_dir))
        return store

    raise ValueError(
        f"Unknown VECTOR_STORE={settings.vector_store!r}. Supported: memory"
    )


def build_english_sources(settings: Settings | None = None) -> EnglishSources:
    """Original English passage text, for display. Absent file is fine."""
    settings = settings or get_settings()
    return EnglishSources(settings.index_dir / _EN_FILE)


def build_chunker(settings: Settings | None = None):
    settings = settings or get_settings()
    provider = settings.chunking_provider.lower()

    if provider == "fixed":
        return FixedSizeChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    if provider == "semantic":
        return SemanticChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    if provider in {"metadata", "metadata_aware"}:
        return MetadataAwareChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    if provider == "recursive":
        return RecursiveChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    if provider in {"parent_child", "parent-child"}:
        return ParentChildChunker(
            child_sentences=settings.child_sentences,
            child_stride=settings.child_stride,
        )
    if provider in {"token_window", "token"}:
        # chunk_size/overlap here are TOKEN counts, not characters.
        return TokenWindowChunker(
            chunk_size=settings.token_chunk_size,
            overlap=settings.token_chunk_overlap,
        )

    supported = ", ".join(sorted(CHUNKING_PRESETS))
    raise ValueError(
        f"Unknown CHUNKING_PROVIDER={settings.chunking_provider!r}. "
        f"Supported: {supported}"
    )


def build_stt(settings: Settings | None = None):
    settings = settings or get_settings()
    provider = (settings.stt_provider or "elevenlabs").lower()

    if provider in {"stub", "none", ""}:
        return StubSTT()
    if provider == "elevenlabs":
        api_key = settings.stt_api_key
        if not api_key:
            return StubSTT()
        base_url = settings.stt_base_url or "https://api.elevenlabs.io/v1"
        return ElevenLabsSTT(
            api_key=api_key,
            model=settings.stt_model,
            base_url=base_url,
        )
    if provider == "openai":
        api_key = settings.stt_api_key or settings.llm_api_key
        base_url = settings.stt_base_url or settings.llm_base_url
        return OpenAIWhisperSTT(
            api_key=api_key,
            model=settings.stt_model,
            base_url=base_url,
        )

    raise ValueError(
        f"Unknown STT_PROVIDER={settings.stt_provider!r}. "
        "Supported: elevenlabs, stub, openai"
    )


def build_bm25_index(settings: Settings | None = None) -> BM25Index:
    """Load the persisted BM25 index, or an empty one if it was never built."""
    settings = settings or get_settings()
    index = BM25Index()
    if settings.index_dir.exists():
        index.load(settings.index_dir)
    return index


def build_retriever(
    settings: Settings | None = None,
    *,
    store=None,
    embedder=None,
    index: BM25Index | None = None,
) -> BaseRetriever:
    """Build the configured retriever.

    Accepts pre-built components so callers that need the store/embedder/BM25
    index anyway (the orchestrator does) can avoid loading a 100k-chunk index
    and a transformer model twice.
    """
    settings = settings or get_settings()
    provider = (settings.retriever_provider or "hybrid").lower()
    store = store if store is not None else build_vector_store(settings)

    if provider == "dense":
        return DenseRetriever(
            embedder=embedder if embedder is not None else build_embedder(settings),
            store=store,
            top_k=settings.top_k,
            dedupe=settings.retrieval_dedupe,
        )
    if provider == "sparse":
        return SparseRetriever(
            index=index if index is not None else build_bm25_index(settings),
            store=store,
            top_k=settings.top_k,
            dedupe=settings.retrieval_dedupe,
        )
    if provider == "hybrid":
        index = index if index is not None else build_bm25_index(settings)
        if index.matrix is None:
            # No lexical index on disk, degrade to dense rather than silently
            # returning nothing, and make the reason visible.
            print(
                "[factory] No BM25 index found in "
                f"{settings.index_dir}; falling back to dense retrieval. "
                "Re-run './hhgoa ingest msmarco' to build it.",
                file=sys.stderr,
            )
            return DenseRetriever(
                embedder=embedder if embedder is not None else build_embedder(settings),
                store=store,
                top_k=settings.top_k,
                dedupe=settings.retrieval_dedupe,
            )
        return HybridRetriever(
            embedder=embedder if embedder is not None else build_embedder(settings),
            store=store,
            index=index,
            top_k=settings.top_k,
            candidate_k=settings.retrieval_candidate_k,
            fusion=settings.fusion_method,
            rrf_k=settings.fusion_rrf_k,
            alpha=settings.fusion_alpha,
            dense_weight=settings.fusion_dense_weight,
            dense_weight_by_language=settings.fusion_dense_weight_by_language,
            dedupe=settings.retrieval_dedupe,
        )

    raise ValueError(
        f"Unknown RETRIEVER={settings.retriever_provider!r}. "
        "Supported: dense, sparse, hybrid"
    )


def build_cross_scorer(settings: Settings | None = None) -> CrossEncoderScorer | None:
    """The gate's cross-encoder, or None when it is switched off."""
    settings = settings or get_settings()
    name = (settings.guardrail_cross_encoder or "").strip()
    return CrossEncoderScorer(name, device=settings.torch_device) if name else None


def _build_grounding_gate(settings: Settings):
    """Multi-feature gate when calibrated, plain cosine threshold otherwise."""
    if (settings.guardrail_mode or "auto").lower() == "threshold":
        return GroundingGate(min_score=settings.guardrail_min_score)
    return ConfidenceGate.from_file(
        settings.guardrail_model_path,
        min_score=settings.guardrail_min_score,
        cross_scorer=build_cross_scorer(settings),
    )


def build_guardrail(settings: Settings | None = None):
    settings = settings or get_settings()
    provider = (settings.guardrail_provider or "default").lower()

    if provider in {"stub", "none", ""}:
        return StubGuardrail()
    if provider == "default":
        return CompositeGuardrail(
            input_filter=InputIntentFilter(
                min_query_length=settings.guardrail_min_query_length,
                supported_languages=settings.supported_languages,
            ),
            grounding_gate=_build_grounding_gate(settings),
            hallucination_checker=HallucinationChecker(
                min_overlap=settings.guardrail_min_answer_overlap,
            ),
        )
    if provider == "input":
        return InputIntentFilter(
            min_query_length=settings.guardrail_min_query_length,
            supported_languages=settings.supported_languages,
        )
    if provider == "grounding":
        return _build_grounding_gate(settings)

    raise ValueError(
        f"Unknown GUARDRAIL_PROVIDER={settings.guardrail_provider!r}. "
        "Supported: default, stub, input, grounding"
    )


_LLM_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    # Groq is OpenAI-compatible, so the same client works with a different host.
    "groq": "https://api.groq.com/openai/v1",
}


def build_chat_clients(settings: Settings | None = None) -> list[ChatClient]:
    """Ordered provider fallback chain for the quality path.

    Returns primary first, then the optional secondary. Unconfigured providers
    are omitted, so an empty list simply means "no quality path available" and
    the harness stays on the fast path.
    """
    settings = settings or get_settings()
    clients: list[ChatClient] = []

    if settings.llm_api_key:
        provider = (settings.llm_provider or "openai").lower()
        clients.append(
            ChatClient(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url or _LLM_BASE_URLS.get(provider, ""),
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                provider=provider,
            )
        )

    if settings.llm_api_key_secondary:
        provider = (settings.llm_provider_secondary or "groq").lower()
        clients.append(
            ChatClient(
                api_key=settings.llm_api_key_secondary,
                model=settings.llm_model_secondary or settings.llm_model,
                base_url=_LLM_BASE_URLS.get(provider, settings.llm_base_url),
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                provider=provider,
            )
        )

    return clients


def build_fast_answerer(
    settings: Settings | None = None,
    *,
    index: BM25Index | None = None,
    embedder=None,
) -> ExtractiveAnswerer:
    """Local extractive answerer, IDF-weighted from the BM25 index when present."""
    settings = settings or get_settings()
    index = index if index is not None else build_bm25_index(settings)
    return ExtractiveAnswerer(
        idf_lookup=index.idf_for if index.idf is not None else None,
        # The retriever's embedder, passed in rather than built: the escalation
        # for mixed-script queries must not load a second copy of the model.
        embedder=embedder if embedder is not None else build_embedder(settings),
    )


def build_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Assemble the harness, loading each shared component exactly once."""
    settings = settings or get_settings()
    store = build_vector_store(settings)
    embedder = build_embedder(settings)
    index = build_bm25_index(settings)
    return Orchestrator(
        retriever=build_retriever(settings, store=store, embedder=embedder, index=index),
        guardrail=build_guardrail(settings),
        fast_answerer=build_fast_answerer(settings, index=index, embedder=embedder),
        chat_clients=build_chat_clients(settings),
        default_language=settings.default_language,
        top_k=settings.top_k,
        answer_cache=AnswerCache(settings.answer_cache_size),
    )


def build_rag_pipeline(
    settings: Settings | None = None,
    *,
    local_only: bool = False,
) -> RAGPipeline:
    """Build the pipeline facade.

    ``local_only`` drops the quality-path clients, so the pipeline cannot make a
    network call. Used by eval and the latency benchmark, where a remote model
    would add variance that has nothing to do with what is being measured.
    """
    settings = settings or get_settings()
    orchestrator = build_orchestrator(settings)
    if local_only:
        orchestrator.chat_clients = []
    return RAGPipeline(orchestrator, default_language=settings.default_language)
