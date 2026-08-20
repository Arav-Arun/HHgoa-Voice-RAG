"""Dense vector retriever: embed the query, search the vector store."""

from __future__ import annotations

from core.embeddings.base import BaseEmbedder
from core.retriever.base import BaseRetriever, dedupe_by_document, fetch_width
from core.types import ScoredChunk
from core.vectorstore.base import BaseVectorStore


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        top_k: int = 5,
        *,
        dedupe: bool = True,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.dedupe = dedupe

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        language: str | None = None,
    ) -> list[ScoredChunk]:
        del language  # dense ranking is language-agnostic
        k = top_k or self.top_k
        embedding = self.embedder.embed_query(query)
        if not self.dedupe:
            return self.store.search(embedding, top_k=k)
        # Overfetch by the index's fan-out so k distinct passages survive the
        # collapse even when one passage owns several of the top chunks.
        candidates = self.store.search(embedding, top_k=fetch_width(k, self.fanout))
        return dedupe_by_document(candidates, k)
