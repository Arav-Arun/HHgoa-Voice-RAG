"""Factory — swap components by editing provider strings in .env or here."""

from __future__ import annotations

from core.chunking.fixed import FixedSizeChunker
from core.config import Settings, get_settings
from core.embeddings.hash import HashEmbedder
from core.embeddings.openai import OpenAIEmbedder
from core.llm.openai_compat import OpenAICompatibleLLM
from core.llm.template import TemplateLLM
from core.rag.pipeline import RAGPipeline
from core.retriever.dense import DenseRetriever
from core.vectorstore.memory import MemoryVectorStore


def build_embedder(settings: Settings | None = None):
    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()

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
        "Supported: hash, openai"
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
    return FixedSizeChunker(
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
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


def build_retriever(settings: Settings | None = None) -> DenseRetriever:
    settings = settings or get_settings()
    embedder = build_embedder(settings)
    store = build_vector_store(settings)
    return DenseRetriever(embedder=embedder, store=store, top_k=settings.top_k)


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
    )
