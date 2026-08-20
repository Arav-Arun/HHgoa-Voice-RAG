"""Sliding window sized in model *tokens* rather than characters.

Why this exists: every other chunker here measures ``chunk_size`` in characters,
and character counts are not comparable across scripts. Devanagari and Gujarati
carry noticeably fewer characters per subword token than Latin under the e5
tokenizer, so a fixed 512-character window yields a materially different
*effective* window per language, a confound sitting underneath every
cross-language chunking comparison in this repo.

Sizing in tokens removes it, and also aligns the window with the encoder's real
constraint, which is its 512-token input limit, not any character count.

The tokenizer is loaded lazily and shared with the embedding model, so this adds
no extra model download.
"""

from __future__ import annotations

from typing import ClassVar

from core.chunking.base import BaseChunker
from core.types import Chunk, Document

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


class TokenWindowChunker(BaseChunker):
    _tokenizers: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        chunk_size: int = 160,
        overlap: int = 24,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        # NOTE: chunk_size/overlap are in TOKENS here, not characters.
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model_name = model_name

    @property
    def tokenizer(self):
        if self.model_name not in self._tokenizers:
            from transformers import AutoTokenizer

            self._tokenizers[self.model_name] = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizers[self.model_name]

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        tokenizer = self.tokenizer
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            return []

        stride = max(self.chunk_size - self.overlap, 1)
        chunks: list[Chunk] = []
        for idx, start in enumerate(range(0, len(ids), stride)):
            window = ids[start : start + self.chunk_size]
            if not window:
                break
            piece = tokenizer.decode(window, skip_special_tokens=True).strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=f"{document.id}__{idx}",
                        text=piece,
                        document_id=document.id,
                        language=document.language,
                        metadata={
                            **document.metadata,
                            "chunking": "token_window",
                            "token_start": start,
                            "token_end": start + len(window),
                        },
                    )
                )
            if start + self.chunk_size >= len(ids):
                break
        return chunks
