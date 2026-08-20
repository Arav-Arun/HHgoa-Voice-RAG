"""Deterministic hash embedder, no API keys, good for local dev."""

from __future__ import annotations

import hashlib
import math
import re

from core.embeddings.base import BaseEmbedder


class HashEmbedder(BaseEmbedder):
    """Feature hashing into a fixed-dim unit vector."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
