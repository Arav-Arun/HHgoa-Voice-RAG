"""Retrieval must rank passages, not chunks.

A chunker with fan-out puts several sibling chunks of one passage into the same
ranking. These tests pin the two ways that used to corrupt the comparison
between chunkers, both described in :mod:`core.retriever.base`.
"""

from __future__ import annotations

import numpy as np

from core.retriever.base import best_per_document, document_fanout, fetch_width
from core.retriever.hybrid import HybridRetriever
from core.retriever.sparse import BM25Index
from core.types import Chunk
from core.vectorstore.memory import MemoryVectorStore


class _FixedEmbedder:
    """Returns one canned query vector, so ranking is fully determined."""

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed_query(self, text: str) -> list[float]:
        del text
        return self.vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


def _store(rows: list[tuple[str, str, list[float]]]) -> MemoryVectorStore:
    store = MemoryVectorStore()
    store.add(
        [Chunk(id=cid, text=cid, document_id=doc) for cid, doc, _ in rows],
        [vec for _, _, vec in rows],
    )
    return store


def _build(store: MemoryVectorStore, texts: list[str], **kwargs) -> HybridRetriever:
    index = BM25Index()
    index.build(texts)
    return HybridRetriever(
        embedder=_FixedEmbedder([1.0, 0.0]),
        store=store,
        index=index,
        candidate_k=10,
        **kwargs,
    )


def test_fanout_and_fetch_width():
    chunks = [Chunk(id=f"c{i}", text="t", document_id=f"d{i // 3}") for i in range(9)]
    assert document_fanout(chunks) == 3.0
    assert document_fanout([]) == 1.0
    # Wide enough that collapsing a 3x fan-out still leaves candidate_k docs.
    assert fetch_width(10, 3.0) == 45
    # Never narrower than the request itself.
    assert fetch_width(10, 1.0) == 15
    assert fetch_width(10, 0.0) == 10


def test_best_per_document_keeps_first_sighting():
    chunks = [
        Chunk(id="a1", text="t", document_id="A"),
        Chunk(id="a2", text="t", document_id="A"),
        Chunk(id="b1", text="t", document_id="B"),
    ]
    # Values are chunk indices. Input is best-first, so A resolves to a1.
    assert best_per_document([0, 1, 2], chunks, limit=5) == {"A": 0, "B": 2}
    # The limit counts documents, not chunks.
    assert best_per_document([0, 1, 2], chunks, limit=1) == {"A": 0}


def test_siblings_do_not_crowd_out_distinct_passages():
    """Three children of one passage must not consume three of five slots."""
    # Passage A owns the three closest vectors; B..E follow.
    rows = [
        ("a1", "A", [0.99, 0.14]),
        ("a2", "A", [0.98, 0.20]),
        ("a3", "A", [0.97, 0.24]),
        ("b1", "B", [0.90, 0.44]),
        ("c1", "C", [0.80, 0.60]),
        ("d1", "D", [0.70, 0.71]),
        ("e1", "E", [0.60, 0.80]),
    ]
    store = _store(rows)
    texts = [cid for cid, _, _ in rows]

    collapsed = _build(store, texts).retrieve("q", top_k=5)
    assert [s.chunk.document_id for s in collapsed] == ["A", "B", "C", "D", "E"]

    raw = _build(store, texts, dedupe=False).retrieve("q", top_k=5)
    assert [s.chunk.document_id for s in raw] == ["A", "A", "A", "B", "C"]


def test_passage_combines_dense_and_lexical_evidence():
    """One passage nominated via different children still fuses both signals.

    Chunk-level fusion scored those children separately, so a passage that both
    retrievers liked got neither the dense nor the lexical rank of the other.
    """
    rows = [
        ("a_dense", "A", [1.0, 0.02]),  # closest vector, no query term
        ("a_lex", "A", [0.10, 0.99]),  # far vector, carries the query term
        ("b1", "B", [0.99, 0.10]),
        ("c1", "C", [0.98, 0.14]),
    ]
    store = _store(rows)
    texts = ["filler one", "kiwi kiwi kiwi", "filler two", "filler three"]

    top = _build(store, texts, dense_weight=0.5).retrieve("kiwi", top_k=1)[0]
    assert top.chunk.document_id == "A"
    # Both ranks present on one candidate is the whole point.
    assert top.components["dense_rank"] == 1.0
    assert top.components["sparse_rank"] == 1.0
    # The encoder's pick represents the passage; its dense score is the max.
    assert top.chunk.id == "a_dense"
    assert top.components["dense_score"] > 0.99
    assert top.components["sparse_score"] > 0.0


def test_zscore_fusion_also_ranks_passages():
    rows = [
        ("a1", "A", [0.99, 0.14]),
        ("a2", "A", [0.98, 0.20]),
        ("b1", "B", [0.90, 0.44]),
    ]
    store = _store(rows)
    results = _build(store, ["x", "y", "z"], fusion="zscore").retrieve("q", top_k=2)
    assert [s.chunk.document_id for s in results] == ["A", "B"]


def test_empty_index_returns_nothing():
    retriever = _build(MemoryVectorStore(), [])
    assert retriever.retrieve("q", top_k=5) == []
    assert retriever.fanout == 1.0


def test_dense_scores_survive_for_the_grounding_gate():
    """The guardrail is calibrated on cosine, so it must not see a fusion score."""
    rows = [("a1", "A", [1.0, 0.0]), ("b1", "B", [0.0, 1.0])]
    store = _store(rows)
    top = _build(store, ["x", "y"]).retrieve("q", top_k=1)[0]
    assert np.isclose(top.dense_score, 1.0)
    assert top.score != top.dense_score
