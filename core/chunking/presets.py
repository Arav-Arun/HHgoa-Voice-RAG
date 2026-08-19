"""Registered chunking strategies — swap via CHUNKING_PROVIDER in .env."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CHUNKING_PROVIDER = "fixed"


@dataclass(frozen=True)
class ChunkingPreset:
    description: str


CHUNKING_PRESETS: dict[str, ChunkingPreset] = {
    "fixed": ChunkingPreset(
        description="Fixed character windows with overlap (baseline)",
    ),
    "semantic": ChunkingPreset(
        description="Sentence-boundary packing up to CHUNK_SIZE (Indic danda + Latin)",
    ),
    "metadata": ChunkingPreset(
        description="Atomic MS MARCO passages when short; semantic fallback when long",
    ),
}
