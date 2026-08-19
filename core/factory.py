"""Factory — swap components by editing provider strings in .env or here."""

from __future__ import annotations

from core.chunking.fixed import FixedSizeChunker
from core.chunking.metadata import MetadataAwareChunker
from core.chunking.presets import CHUNKING_PRESETS, DEFAULT_CHUNKING_PROVIDER
from core.chunking.semantic import SemanticChunker
from core.config import Settings, get_settings
from core.embeddings.hash import HashEmbedder
from core.embeddings.openai import OpenAIEmbedder
from core.embeddings.presets import DEFAULT_EMBEDDING_PRESET, EMBEDDING_PRESETS
from core.embeddings.sentence_transformers import SentenceTransformerEmbedder
from core.guardrails.composite import CompositeGuardrail
from core.guardrails.grounding import GroundingGate
from core.guardrails.hallucination import HallucinationChecker
from core.guardrails.input_intent import InputIntentFilter
from core.guardrails.stub import StubGuardrail
from core.llm.openai_compat import OpenAICompatibleLLM
from core.llm.template import TemplateLLM
from core.rag.pipeline import RAGPipeline
from core.retriever.dense import DenseRetriever
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
        return SentenceTransformerEmbedder(
            model_name=model_name,
            query_prefix=preset.query_prefix if preset else "",
            passage_prefix=preset.passage_prefix if preset else "",
            batch_size=settings.embedding_batch_size,
        )
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

    supported = ", ".join(sorted(CHUNKING_PRESETS))
    raise ValueError(
        f"Unknown CHUNKING_PROVIDER={settings.chunking_provider!r}. "
        f"Supported: {supported}"
    )


def build_llm(settings: Settings | None = None, *, use_template: bool = False):
    settings = settings or get_settings()
    if use_template or not settings.llm_api_key:
        return TemplateLLM()
    return OpenAICompatibleLLM(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
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


def build_retriever(settings: Settings | None = None) -> DenseRetriever:
    settings = settings or get_settings()
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    return DenseRetriever(embedder=embedder, store=store, top_k=settings.top_k)


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
            grounding_gate=GroundingGate(min_score=settings.guardrail_min_score),
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
        return GroundingGate(min_score=settings.guardrail_min_score)

    raise ValueError(
        f"Unknown GUARDRAIL_PROVIDER={settings.guardrail_provider!r}. "
        "Supported: default, stub, input, grounding"
    )


def build_rag_pipeline(
    settings: Settings | None = None,
    *,
    use_template_llm: bool = False,
) -> RAGPipeline:
    settings = settings or get_settings()
    return RAGPipeline(
        retriever=build_retriever(settings),
        llm=build_llm(settings, use_template=use_template_llm),
        default_language=settings.default_language,
        guardrail=build_guardrail(settings),
    )
