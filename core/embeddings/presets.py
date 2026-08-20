"""Known local embedding presets, swap via EMBEDDING_PRESET in .env."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingPreset:
    model: str
    query_prefix: str = ""
    passage_prefix: str = ""
    description: str = ""


EMBEDDING_PRESETS: dict[str, EmbeddingPreset] = {
    "e5-small": EmbeddingPreset(
        model="intfloat/multilingual-e5-small",
        query_prefix="query: ",
        passage_prefix="passage: ",
        description="Multilingual E5-small, fast default for hi/gu retrieval",
    ),
    "indic-sbert": EmbeddingPreset(
        model="l3cube-pune/indic-sentence-similarity-sbert",
        description=(
            "IndicSBERT (MuRIL-based), Indic-focused comparison model; "
            "ai4bharat has no dedicated retrieval encoder yet"
        ),
    ),
    "bge-m3": EmbeddingPreset(
        model="BAAI/bge-m3",
        query_prefix="Represent this sentence for searching relevant passages: ",
        description="BGE-M3, stronger multilingual fallback, larger/slower",
    ),
}

DEFAULT_EMBEDDING_PRESET = "e5-small"
