"""Central configuration — override via .env or tweak defaults here."""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.embeddings.presets import DEFAULT_EMBEDDING_PRESET
from core.chunking.presets import DEFAULT_CHUNKING_PROVIDER


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

    # Chunking — swap strategy in core/chunking/
    chunking_provider: str = Field(
        default=DEFAULT_CHUNKING_PROVIDER,
        validation_alias="CHUNKING_PROVIDER",
    )
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 5
    embedding_dim: int = 384  # hash embedder only; ST models infer dimension

    # Embeddings — sentence_transformers (local) | hash | openai
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

    # Vector store — set VECTOR_STORE=memory|faiss (faiss stub for now)
    vector_store: str = "memory"

    # LLM — OpenAI-compatible chat completions
    llm_api_key: str = Field(default="", validation_alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.openai.com/v1", validation_alias="LLM_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", validation_alias="LLM_MODEL")
    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # STT — speech-to-text for voice queries (hi | gu); spec requires Sarvam or ElevenLabs
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

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000


def get_settings() -> Settings:
    return Settings()
