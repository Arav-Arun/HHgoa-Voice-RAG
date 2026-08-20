"""Central configuration, override via .env or tweak defaults here."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.chunking.presets import DEFAULT_CHUNKING_PROVIDER
from core.embeddings.presets import DEFAULT_EMBEDDING_PRESET


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Path("data")
    index_dir: Path = Path("data/index")
    samples_dir: Path = Path("data/samples")

    # Scope defaults (Hindi + Gujarati)
    default_language: str = "hi"
    supported_languages: tuple[str, ...] = ("hi", "gu")

    # Chunking, swap strategy in core/chunking/
    chunking_provider: str = Field(
        default=DEFAULT_CHUNKING_PROVIDER,
        validation_alias="CHUNKING_PROVIDER",
    )
    chunk_size: int = 512
    chunk_overlap: int = 64
    # parent_child: children are this many sentences, advanced by this stride.
    # 3 is the dev-selected window (`./hhgoa child-sweep`). A per-language
    # window looked justified on eval and did not replicate on dev, so there
    # is deliberately only one knob here.
    child_sentences: int = Field(default=3, validation_alias="CHILD_SENTENCES")
    child_stride: int = Field(default=1, validation_alias="CHILD_STRIDE")
    # token_window sizes are in e5 TOKENS, not characters.
    token_chunk_size: int = Field(default=160, validation_alias="TOKEN_CHUNK_SIZE")
    token_chunk_overlap: int = Field(default=24, validation_alias="TOKEN_CHUNK_OVERLAP")

    # Retrieval
    top_k: int = 5
    embedding_dim: int = 384  # hash embedder only; ST models infer dimension

    # Retriever, dense (embeddings only) | sparse (BM25 only) | hybrid (fused)
    retriever_provider: str = Field(default="hybrid", validation_alias="RETRIEVER")
    # Candidates pulled from each retriever before fusion; only top_k survive.
    retrieval_candidate_k: int = Field(default=50, validation_alias="RETRIEVAL_CANDIDATE_K")
    # Collapse sibling chunks of the same passage before truncating to top_k.
    # Set false only to reproduce the pre-dedup ablation.
    retrieval_dedupe: bool = Field(default=True, validation_alias="RETRIEVAL_DEDUPE")
    # rrf (rank-based, scale-free) | zscore (score-based, needs alpha tuning)
    fusion_method: str = Field(default="rrf", validation_alias="FUSION")
    fusion_rrf_k: int = Field(default=60, validation_alias="RRF_K")
    fusion_alpha: float = Field(default=0.5, validation_alias="FUSION_ALPHA")
    # Weighted-RRF dense share. Tuned on the dev slice (data/eval/fusion-sweep.json):
    # Hindi peaks at 0.95 (dense encoder is strong, lexical mostly adds noise),
    # Gujarati at 0.50 (dense is weak there, BM25 genuinely rescues it).
    fusion_dense_weight: float = Field(default=0.8, validation_alias="FUSION_DENSE_WEIGHT")
    fusion_dense_weight_hi: float = Field(default=0.95, validation_alias="FUSION_DENSE_WEIGHT_HI")
    fusion_dense_weight_gu: float = Field(default=0.50, validation_alias="FUSION_DENSE_WEIGHT_GU")

    @property
    def fusion_dense_weight_by_language(self) -> dict[str, float]:
        return {"hi": self.fusion_dense_weight_hi, "gu": self.fusion_dense_weight_gu}

    # Embeddings, sentence_transformers (local) | hash | openai
    embedding_provider: str = Field(
        default="sentence_transformers",
        validation_alias="EMBEDDING_PROVIDER",
    )
    embedding_preset: str = Field(
        default=DEFAULT_EMBEDDING_PRESET,
        validation_alias="EMBEDDING_PRESET",
    )
    embedding_model: str = Field(default="", validation_alias="EMBEDDING_MODEL")
    embedding_batch_size: int = Field(default=32, validation_alias="EMBEDDING_BATCH_SIZE")
    # Torch device for the query encoder and the gate's cross-encoder. Both
    # models are small and run one item at a time, where GPU dispatch overhead
    # costs more than it saves: measured 7.46 ms on MPS against 5.32 ms on CPU.
    # Empty string hands the choice back to sentence-transformers.
    torch_device: str = Field(default="cpu", validation_alias="TORCH_DEVICE")

    # Vector store, set VECTOR_STORE=memory|faiss (faiss stub for now)
    vector_store: str = "memory"

    # LLM, OpenAI-compatible chat completions
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", validation_alias="LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")
    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # Optional secondary provider, the harness fails over to this when the
    # primary errors. Groq is OpenAI-compatible, so only the host/model differ.
    llm_api_key_secondary: str = Field(default="", validation_alias="LLM_API_KEY_SECONDARY")
    llm_provider_secondary: str = Field(default="groq", validation_alias="LLM_PROVIDER_SECONDARY")
    llm_model_secondary: str = Field(default="", validation_alias="LLM_MODEL_SECONDARY")

    # STT, speech-to-text for voice queries (hi | gu); spec requires Sarvam or ElevenLabs
    stt_provider: str = Field(default="elevenlabs", validation_alias="STT_PROVIDER")
    stt_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("STT_API_KEY", "ELEVENLABS_API_KEY"),
    )
    stt_base_url: str = Field(
        default="https://api.elevenlabs.io/v1",
        validation_alias="STT_BASE_URL",
    )
    stt_model: str = Field(default="scribe_v2", validation_alias="STT_MODEL")

    # Guardrails, input intent filter + grounding/abstain gate
    guardrail_provider: str = Field(default="default", validation_alias="GUARDRAIL_PROVIDER")
    guardrail_min_score: float = Field(default=0.86, validation_alias="GUARDRAIL_MIN_SCORE")
    guardrail_min_query_length: int = Field(default=3, validation_alias="GUARDRAIL_MIN_QUERY_LENGTH")
    guardrail_min_answer_overlap: float = Field(
        default=0.20,
        validation_alias="GUARDRAIL_MIN_ANSWER_OVERLAP",
    )
    # auto: use the fitted multi-feature gate when its model file exists.
    # threshold: force the plain cosine threshold (ablation / fresh checkout).
    guardrail_mode: str = Field(default="auto", validation_alias="GUARDRAIL_MODE")
    guardrail_model_path: Path = Field(
        default=Path("data/eval/guardrail-model.json"),
        validation_alias="GUARDRAIL_MODEL_PATH",
    )
    # Cross-encoder relevance check for the grounding gate. Reads the query and
    # the top passage together, which is the only feature that catches a
    # passage about the wrong variant of the right subject. Empty disables it.
    guardrail_cross_encoder: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        validation_alias="GUARDRAIL_CROSS_ENCODER",
    )

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000


def get_settings() -> Settings:
    return Settings()
