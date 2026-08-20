"""Registered chunking strategies, swap via CHUNKING_PROVIDER in .env."""

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
    "recursive": ChunkingPreset(
        description="Separator cascade: paragraph -> danda -> clause -> char, with overlap",
    ),
    "parent_child": ChunkingPreset(
        description="Embed 2-sentence children for precision; document_id stays the parent",
    ),
    "token_window": ChunkingPreset(
        description="Sliding window sized in e5 tokens, not characters (script-fair)",
    ),
}
