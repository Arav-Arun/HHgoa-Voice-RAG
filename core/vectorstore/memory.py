"""In-memory vector store with JSON + numpy persistence.

Hot-path note: vectors are L2-normalized **once**, at ``add()``/``load()`` time,
so a query is a single matrix-vector product. The previous implementation
renormalized the entire matrix on every ``search()`` call, which allocated a
full copy of the embedding matrix per query (~150 MB at 100k x 384 float32) and
dominated retrieval latency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.types import Chunk, ScoredChunk
from core.vectorstore.base import BaseVectorStore


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class MemoryVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # always L2-normalized

    @property
    def chunks(self) -> list[Chunk]:
        return self._chunks

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        self._chunks.extend(chunks)
        new_matrix = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        if self._matrix is None:
            self._matrix = new_matrix
        else:
            self._matrix = np.vstack([self._matrix, new_matrix])

    def score_all(self, query_embedding: list[float]) -> np.ndarray:
        """Cosine similarity against every chunk.

        Exposed separately so hybrid retrieval can reuse a single matmul for
        both ranking and for back-filling calibrated dense scores onto
        candidates that only the sparse retriever surfaced.
        """
        if self._matrix is None or not self._chunks:
            return np.zeros(0, dtype=np.float32)

        query = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return np.zeros(len(self._chunks), dtype=np.float32)

        # Matrix rows are pre-normalized, so this dot product is cosine similarity.
        return self._matrix @ (query / query_norm)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            return []

        # A zero-norm query carries no direction; ranking it is meaningless.
        if float(np.linalg.norm(np.asarray(query_embedding, dtype=np.float32))) == 0.0:
            return []

        scores = self.score_all(query_embedding)
        if scores.size == 0:
            return []
        k = min(top_k, len(self._chunks))

        # argpartition is O(n) vs argsort's O(n log n); only the k winners get sorted.
        top_unsorted = np.argpartition(scores, -k)[-k:]
        top_indices = top_unsorted[np.argsort(scores[top_unsorted])[::-1]]

        return [
            ScoredChunk(
                chunk=self._chunks[i],
                score=float(scores[i]),
                components={"dense_score": float(scores[i]), "dense_rank": float(rank)},
            )
            for rank, i in enumerate(top_indices, start=1)
        ]

    def save(self, path: str | None = None) -> None:
        if path is None:
            raise ValueError("path is required for MemoryVectorStore.save")

        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        chunks_path = target / "chunks.json"
        matrix_path = target / "embeddings.npy"

        payload = [
            {
                "id": c.id,
                "text": c.text,
                "document_id": c.document_id,
                "language": c.language,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        chunks_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if self._matrix is not None:
            np.save(matrix_path, self._matrix)
        elif matrix_path.exists():
            matrix_path.unlink()

    def load(self, path: str | None = None) -> None:
        if path is None:
            raise ValueError("path is required for MemoryVectorStore.load")

        target = Path(path)
        chunks_path = target / "chunks.json"
        matrix_path = target / "embeddings.npy"

        if not chunks_path.exists():
            self._chunks = []
            self._matrix = None
            return

        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._chunks = [
            Chunk(
                id=row["id"],
                text=row["text"],
                document_id=row["document_id"],
                language=row.get("language", "hi"),
                metadata=row.get("metadata", {}),
            )
            for row in raw
        ]
        # Re-normalize on load: indexes written by older builds stored raw vectors.
        self._matrix = _l2_normalize(np.load(matrix_path)) if matrix_path.exists() else None

    def count(self) -> int:
        return len(self._chunks)
