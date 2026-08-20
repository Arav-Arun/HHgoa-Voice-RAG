"""Local sentence-transformers embedder with query/passage prefix support."""

from __future__ import annotations

from typing import ClassVar

from core.embeddings.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps Hugging Face sentence-transformers models for dense retrieval."""

    _models: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        model_name: str,
        *,
        query_prefix: str = "",
        passage_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        if not model_name:
            raise ValueError("model_name is required for SentenceTransformerEmbedder")
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size
        self._dimension: int | None = None

    @property
    def model(self):
        if self.model_name not in self._models:
            from sentence_transformers import SentenceTransformer

            self._models[self.model_name] = SentenceTransformer(self.model_name)
        return self._models[self.model_name]

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
