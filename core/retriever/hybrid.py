"""Hybrid retrieval: dense embeddings + BM25, fused at the passage level.

Why fuse by *rank* rather than score by default: e5 cosine similarities on this
corpus occupy a narrow, high band (empirically ~0.79-0.96) while BM25 is
unbounded and corpus-dependent. Any fixed weighting of the two raw scales is
dominated by BM25's larger dynamic range. Reciprocal Rank Fusion is scale-free
and needs no per-corpus tuning, so it is the default; weighted z-score fusion is
available for comparison and its alpha is swept on the **dev** slice only.

Fusion operates on *passages*, not chunks. Each retriever's candidate list is
collapsed to its best chunk per ``document_id`` before ranks are assigned, for
the reasons in :mod:`core.retriever.base`. Collapsing also lets one passage
combine its dense and lexical evidence: nominated as child A by the encoder and
child B by BM25, it is still one candidate carrying both signals, where
chunk-level fusion would score the two children separately and give neither the
sum.

Both retrievers' signals survive onto every result in ``ScoredChunk.components``
so the grounding guardrail can read the calibrated dense cosine rather than the
fusion score, which has no absolute meaning.
"""

from __future__ import annotations

import numpy as np

from core.embeddings.base import BaseEmbedder
from core.retriever.base import BaseRetriever, best_per_document, fetch_width
from core.retriever.sparse import BM25Index
from core.types import ScoredChunk
from core.vectorstore.base import BaseVectorStore

DEFAULT_RRF_K = 60
DEFAULT_CANDIDATE_K = 50


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores, best first."""
    if scores.size == 0:
        return np.empty(0, dtype=int)
    k = min(k, scores.size)
    candidates = np.argpartition(scores, -k)[-k:]
    return candidates[np.argsort(scores[candidates])[::-1]]


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        index: BM25Index,
        *,
        top_k: int = 5,
        candidate_k: int = DEFAULT_CANDIDATE_K,
        fusion: str = "rrf",
        rrf_k: int = DEFAULT_RRF_K,
        alpha: float = 0.5,
        dense_weight: float = 0.5,
        dense_weight_by_language: dict[str, float] | None = None,
        dedupe: bool = True,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.index = index
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.alpha = alpha
        # Weighted RRF. A single global weight is the wrong abstraction here:
        # measured on the dev slice, the dense encoder is strong on Hindi and
        # weak on Gujarati, so the optimal dense/lexical balance differs by
        # language. `dense_weight_by_language` overrides the global default.
        self.dense_weight = dense_weight
        self.dense_weight_by_language = dense_weight_by_language or {}
        # False reproduces naive chunk-level fusion, the pre-fix ablation.
        self.dedupe = dedupe

    def weight_for(self, language: str | None) -> float:
        return self.dense_weight_by_language.get(language or "", self.dense_weight)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        language: str | None = None,
    ) -> list[ScoredChunk]:
        k = top_k or self.top_k
        chunks = self._chunks()
        if not chunks:
            return []

        # One embedding + one matmul serves both ranking and the dense
        # back-fill for sparse-only candidates.
        dense_scores = self.store.score_all(self.embedder.embed_query(query))
        sparse_scores = self.index.score(query)
        if dense_scores.size == 0:
            dense_scores = np.zeros(len(chunks), dtype=np.float32)
        if sparse_scores.size == 0:
            sparse_scores = np.zeros(len(chunks), dtype=np.float32)

        # Candidates are keyed by passage when deduping and by chunk when not,
        # so the same fusion below serves both the fix and its ablation.
        width = fetch_width(self.candidate_k, self.fanout) if self.dedupe else self.candidate_k
        dense_pool = [int(i) for i in _top_indices(dense_scores, width)]
        # Only positive BM25 scores are real lexical matches; zeros are misses.
        sparse_pool = [int(i) for i in _top_indices(sparse_scores, width) if sparse_scores[i] > 0]

        if self.dedupe:
            dense_pick = best_per_document(dense_pool, chunks, self.candidate_k)
            sparse_pick = best_per_document(sparse_pool, chunks, self.candidate_k)
        else:
            # Every chunk is its own group, i.e. no collapsing at all.
            dense_pick = {chunks[i].id: i for i in dense_pool[: self.candidate_k]}
            sparse_pick = {chunks[i].id: i for i in sparse_pool[: self.candidate_k]}

        dense_rank = {key: rank for rank, key in enumerate(dense_pick, start=1)}
        sparse_rank = {key: rank for rank, key in enumerate(sparse_pick, start=1)}
        candidates = sorted(set(dense_rank) | set(sparse_rank))
        if not candidates:
            return []

        # A passage is represented by the encoder's pick when it has one: that
        # is its semantically closest child, and so the best context to hand the
        # answerer. Membership test rather than truthiness, since index 0 is a
        # valid pick.
        pick = {c: dense_pick[c] if c in dense_pick else sparse_pick[c] for c in candidates}
        # Read each signal off the child that maximised it, so a passage keeps
        # its true dense cosine even when BM25 nominated a different child.
        dense_of = {c: float(dense_scores[dense_pick.get(c, pick[c])]) for c in candidates}
        sparse_of = {c: float(sparse_scores[sparse_pick.get(c, pick[c])]) for c in candidates}

        if self.fusion == "rrf":
            w_dense = self.weight_for(language)
            w_sparse = 1.0 - w_dense
            fused = {
                c: (w_dense / (self.rrf_k + dense_rank[c]) if c in dense_rank else 0.0)
                + (w_sparse / (self.rrf_k + sparse_rank[c]) if c in sparse_rank else 0.0)
                for c in candidates
            }
        elif self.fusion in {"zscore", "weighted"}:
            fused = self._zscore_fusion(candidates, dense_of, sparse_of)
        else:
            raise ValueError(f"Unknown fusion={self.fusion!r}. Supported: rrf, zscore")

        ranked = sorted(candidates, key=lambda c: fused[c], reverse=True)[:k]

        results: list[ScoredChunk] = []
        for key in ranked:
            components = {
                "dense_score": dense_of[key],
                "sparse_score": sparse_of[key],
                "fused_score": float(fused[key]),
            }
            if key in dense_rank:
                components["dense_rank"] = float(dense_rank[key])
            if key in sparse_rank:
                components["sparse_rank"] = float(sparse_rank[key])
            results.append(
                ScoredChunk(chunk=chunks[pick[key]], score=float(fused[key]), components=components)
            )
        return results

    def _zscore_fusion(
        self,
        candidates: list[str],
        dense_of: dict[str, float],
        sparse_of: dict[str, float],
    ) -> dict[str, float]:
        """Standardize each signal across the candidate pool, then blend."""

        def standardize(values: np.ndarray) -> np.ndarray:
            std = float(values.std())
            if std == 0.0:
                return np.zeros_like(values)
            return (values - float(values.mean())) / std

        z_dense = standardize(np.array([dense_of[c] for c in candidates], dtype=np.float32))
        z_sparse = standardize(np.array([sparse_of[c] for c in candidates], dtype=np.float32))
        blended = self.alpha * z_dense + (1.0 - self.alpha) * z_sparse
        return {c: float(v) for c, v in zip(candidates, blended, strict=True)}
