"""Retriever interface and the parent-level candidate handling they share.

Every chunker sets ``Chunk.document_id`` to the source passage, so a chunker
with fan-out (``parent_child`` emits ~3 children per passage) puts several
sibling chunks into one ranking. Two things then go wrong, and both are
properties of the plumbing rather than of the strategy:

* **Rank contamination.** Reciprocal Rank Fusion scores ``1/(k + rank)``. Three
  siblings holding ranks 1-3 push the next distinct passage to rank 4, so a
  fan-out chunker's second-best passage is scored as though it were fourth. The
  penalty scales with fan-out and says nothing about retrieval quality.
* **Pool starvation.** A candidate budget counted in *chunks* buys fewer
  distinct passages the more a chunker splits. Measured on this corpus, a
  50-chunk pool holds 47.8 distinct passages under ``fixed`` but 35.3 under
  ``parent_child``.

So candidates collapse to one best chunk per passage *before* ranks are
assigned, and the budget is counted in passages. ``candidate_k`` then means the
same thing for every chunker, which is what makes the comparison fair.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from core.types import Chunk, ScoredChunk

# Chunks fetched per candidate slot, as a multiple of the index's fan-out.
# Fan-out is a corpus mean, and the head of a similarity ranking clusters
# siblings harder than the mean does, so the raw multiple undershoots. 1.5x
# covers the gap in one pass; the cost is a wider argpartition, which is O(n)
# in the corpus either way.
FETCH_MARGIN = 1.5


def document_fanout(chunks: Sequence[Chunk]) -> float:
    """Mean chunks per source document. 1.0 for a chunker that never splits."""
    if not chunks:
        return 1.0
    return len(chunks) / len({chunk.document_id for chunk in chunks})


def fetch_width(candidate_k: int, fanout: float) -> int:
    """Chunks to pull so that collapsing still leaves ``candidate_k`` documents."""
    return max(candidate_k, math.ceil(candidate_k * fanout * FETCH_MARGIN))


def best_per_document(
    ranked: Iterable[int],
    chunks: Sequence[Chunk],
    limit: int,
) -> dict[str, int]:
    """Map each source document to its best chunk, from a best-first ranking.

    The input is already sorted, so the first sighting of a document is its
    strongest chunk and the returned insertion order is rank order.
    """
    best: dict[str, int] = {}
    for idx in ranked:
        best.setdefault(chunks[idx].document_id, int(idx))
        if len(best) >= limit:
            break
    return best


def dedupe_by_document(scored: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Keep the best-scoring chunk per source document, preserving rank order.

    The score-ranked counterpart of :func:`best_per_document`, for retrievers
    that rank on one signal and so never need parent-level fusion.
    """
    best: dict[str, ScoredChunk] = {}
    for item in scored:
        best.setdefault(item.chunk.document_id, item)
        if len(best) >= top_k:
            break
    return list(best.values())


class BaseRetriever(ABC):
    _fanout: float | None = None

    def _chunks(self) -> list[Chunk]:
        return getattr(self.store, "chunks", [])

    @property
    def fanout(self) -> float:
        """Mean chunks per source document, measured once per index load.

        Cached because it walks the whole index. At 300k chunks that is far too
        slow to repeat inside a 200ms query budget.
        """
        if self._fanout is None:
            self._fanout = document_fanout(self._chunks())
        return self._fanout

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        language: str | None = None,
    ) -> list[ScoredChunk]:
        """Return ranked chunks for a query, best first.

        ``language`` is an optional hint. Implementations that rank identically
        for every language ignore it; hybrid fusion uses it to pick the
        dense/lexical balance, which differs measurably between Hindi and
        Gujarati.
        """
