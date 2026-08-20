"""BM25 lexical retrieval over a precomputed sparse matrix.

Dense embeddings are weak exactly where lexical matching is strong: rare
entities, transliterated names, product codes, and numbers. This is the sparse
half of the hybrid retriever.

Design note, the BM25 document weight

    w[d,t] = idf[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

is entirely **query-independent**, so it is computed once at index time and
stored as a CSC matrix. A query then costs one column lookup per query term
plus a vectorized add, rather than rescoring the corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

from core.retriever.base import BaseRetriever, dedupe_by_document, fetch_width
from core.text import tokenize
from core.types import ScoredChunk
from core.vectorstore.base import BaseVectorStore

BM25_K1 = 1.2
BM25_B = 0.75

_MATRIX_FILE = "bm25.npz"
_VOCAB_FILE = "bm25_vocab.json"
_IDF_FILE = "bm25_idf.npy"


class BM25Index:
    """Sparse BM25 index aligned row-for-row with the vector store's chunks."""

    def __init__(
        self,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.matrix: sparse.csc_matrix | None = None
        self.idf: np.ndarray | None = None
        self.n_docs = 0

    # ---------------------------------------------------------------- build

    def build(self, texts: list[str]) -> None:
        """Build the weighted term-document matrix from chunk texts."""
        self.n_docs = len(texts)
        if not texts:
            self.matrix = None
            self.vocab = {}
            self.idf = None
            return

        doc_counts: list[Counter[str]] = []
        doc_lengths = np.zeros(self.n_docs, dtype=np.float32)
        vocab: dict[str, int] = {}

        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            doc_counts.append(counts)
            doc_lengths[i] = sum(counts.values())
            for term in counts:
                if term not in vocab:
                    vocab[term] = len(vocab)

        self.vocab = vocab
        n_terms = len(vocab)
        avgdl = float(doc_lengths.mean()) or 1.0

        # Document frequency per term, for idf.
        df = np.zeros(n_terms, dtype=np.float32)
        for counts in doc_counts:
            for term in counts:
                df[vocab[term]] += 1.0

        # Lucene-style idf: always positive, so a term in every document
        # contributes ~0 rather than a negative score.
        idf = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # Kept for reuse by the extractive answerer, which weights candidate
        # sentences by how informative their matched query terms are.
        self.idf = idf

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for i, counts in enumerate(doc_counts):
            # Length normalization denominator is per-document, so hoist it.
            norm = self.k1 * (1.0 - self.b + self.b * doc_lengths[i] / avgdl)
            for term, tf in counts.items():
                t = vocab[term]
                rows.append(i)
                cols.append(t)
                data.append(float(idf[t] * tf * (self.k1 + 1.0) / (tf + norm)))

        self.matrix = sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(self.n_docs, n_terms),
        ).tocsc()

    # ---------------------------------------------------------------- score

    def score(self, query: str) -> np.ndarray:
        """BM25 score for every document. Zeros when nothing matches."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if self.matrix is None or not self.vocab:
            return scores

        query_terms = Counter(tokenize(query))
        indptr, indices, data = self.matrix.indptr, self.matrix.indices, self.matrix.data

        for term, qtf in query_terms.items():
            t = self.vocab.get(term)
            if t is None:
                continue
            start, end = indptr[t], indptr[t + 1]
            if start == end:
                continue
            # Row indices within a single CSC column are unique, so this
            # fancy-indexed accumulation is safe (no np.add.at needed).
            scores[indices[start:end]] += data[start:end] * qtf

        return scores

    def top_k(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.score(query)
        if self.n_docs == 0:
            return []
        k = min(k, self.n_docs)
        candidates = np.argpartition(scores, -k)[-k:]
        ordered = candidates[np.argsort(scores[candidates])[::-1]]
        return [(int(i), float(scores[i])) for i in ordered if scores[i] > 0.0]

    # ---------------------------------------------------------------- io

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if self.matrix is None:
            return
        sparse.save_npz(target / _MATRIX_FILE, self.matrix)
        if self.idf is not None:
            np.save(target / _IDF_FILE, self.idf)
        (target / _VOCAB_FILE).write_text(
            json.dumps(
                {"vocab": self.vocab, "n_docs": self.n_docs, "k1": self.k1, "b": self.b},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> bool:
        """Load a persisted index. Returns False when none is present."""
        target = Path(path)
        matrix_path = target / _MATRIX_FILE
        vocab_path = target / _VOCAB_FILE
        if not matrix_path.exists() or not vocab_path.exists():
            return False

        meta = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.vocab = meta["vocab"]
        self.n_docs = int(meta["n_docs"])
        self.k1 = float(meta.get("k1", BM25_K1))
        self.b = float(meta.get("b", BM25_B))
        self.matrix = sparse.load_npz(matrix_path).tocsc()
        idf_path = target / _IDF_FILE
        self.idf = np.load(idf_path) if idf_path.exists() else None
        return True

    def idf_for(self, term: str) -> float:
        """Inverse document frequency for a term; 0.0 if unseen at index time."""
        if self.idf is None:
            return 0.0
        t = self.vocab.get(term)
        return float(self.idf[t]) if t is not None else 0.0


class SparseRetriever(BaseRetriever):
    """BM25-only retriever, mainly useful as an ablation against hybrid."""

    def __init__(
        self,
        index: BM25Index,
        store: BaseVectorStore,
        top_k: int = 5,
        *,
        dedupe: bool = True,
    ) -> None:
        self.index = index
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
        del language  # BM25 ranking is language-agnostic
        k = top_k or self.top_k
        chunks = self._chunks()
        fetch = fetch_width(k, self.fanout) if self.dedupe else k

        results: list[ScoredChunk] = []
        for rank, (idx, score) in enumerate(self.index.top_k(query, fetch), start=1):
            if idx >= len(chunks):
                continue
            results.append(
                ScoredChunk(
                    chunk=chunks[idx],
                    score=score,
                    components={"sparse_score": score, "sparse_rank": float(rank)},
                )
            )
        return dedupe_by_document(results, k) if self.dedupe else results
