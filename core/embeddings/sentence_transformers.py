"""Local sentence-transformers embedder with query/passage prefix support.

Device is chosen explicitly rather than left to sentence-transformers' default.
e5-small is small and queries arrive one at a time, so the per-call overhead of
dispatching to a GPU exceeds what the parallelism buys: measured on Apple
Silicon, one query embedding takes 7.46 ms on MPS and 5.32 ms on CPU. CPU is
also what a deployment box has, so it is the default here and the numbers in
docs/latency.md are the ones a deploy will reproduce.
"""

from __future__ import annotations

from typing import ClassVar

from core.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps Hugging Face sentence-transformers models for dense retrieval."""

    _models: ClassVar[dict[tuple[str, str | None], object]] = {}

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
        device: str = "cpu",
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required for SentenceTransformerEmbedder")
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        # Empty string means "let sentence-transformers decide".
        self.device = device or None
        self._dimension: int | None = None

    @property
    def model(self):
        key = (self.model_name, self.device)
        if key not in self._models:
            from sentence_transformers import SentenceTransformer

            self._models[key] = SentenceTransformer(self.model_name, device=self.device)
        return self._models[key]

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = int(self.model.get_sentence_embedding_dimension())
        return self._dimension

    def embed_texts(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []

        prefix = self.query_prefix if is_query else self.passage_prefix
        inputs = [f"{prefix}{text}" if prefix else text for text in texts]
        vectors = self.model.encode(
            inputs,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query], is_query=True)[0]
