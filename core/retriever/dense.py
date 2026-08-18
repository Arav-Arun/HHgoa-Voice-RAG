"""Dense vector retriever — embed query, search vector store."""

from __future__ import annotations

from core.embeddings.base import BaseEmbedder
from core.retriever.base import BaseRetriever
from core.types import ScoredChunk
from core.vectorstore.base import BaseVectorStore


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        k = top_k or self.top_k
        embedding = self.embedder.embed_query(query)
        return self.store.search(embedding, top_k=k)
