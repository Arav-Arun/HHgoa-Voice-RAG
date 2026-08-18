"""In-memory vector store with JSON + numpy persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.types import Chunk, ScoredChunk
from core.vectorstore.base import BaseVectorStore


class MemoryVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return

        self._chunks.extend(chunks)
        new_matrix = np.array(embeddings, dtype=np.float32)
        if self._matrix is None:
            self._matrix = new_matrix
        else:
            self._matrix = np.vstack([self._matrix, new_matrix])

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        query = query / query_norm
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized = self._matrix / norms
        scores = normalized @ query
        k = min(top_k, len(self._chunks))
        top_indices = np.argsort(scores)[::-1][:k]

        return [
            ScoredChunk(chunk=self._chunks[i], score=float(scores[i]))
            for i in top_indices
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
        chunks_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
        self._matrix = np.load(matrix_path) if matrix_path.exists() else None

    def count(self) -> int:
        return len(self._chunks)
