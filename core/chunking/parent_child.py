"""Parent-child (small-to-big) chunking.

Retrieval precision and answer context want different chunk sizes. A short
window embeds into a sharper vector, less topic dilution, so the query matches
the specific claim rather than the passage's general subject, while an answer
still wants the surrounding sentences for context.

So: embed *child* windows of one or two sentences, but keep ``document_id``
pointing at the parent passage. Retrieval therefore ranks on the sharp child
vector while every downstream consumer (eval labels, citations, the extractive
answerer) continues to resolve to the parent passage. Nothing else in the
pipeline needs to know this chunker was used.

Parent text is intentionally *not* duplicated into each child's metadata: at
100k+ chunks that multiplies index size by the child fan-out, and the parent is
recoverable by grouping on ``document_id``.

This chunker only works if retrieval ranks *passages*. Siblings otherwise crowd
each other out of the top-k and the fan-out reads as a quality loss; see
:mod:`core.retriever.base`, which collapses candidates before ranking them.

Sizing is in sentences, so ``chunk_size``/``chunk_overlap`` do not apply.
"""

from __future__ import annotations

from core.chunking.base import BaseChunker
from core.chunking.sentences import split_sentences
from core.types import Chunk, Document


class ParentChildChunker(BaseChunker):
    def __init__(self, child_sentences: int = 3, child_stride: int = 1) -> None:
        # Window size is chosen on the dev slice by `./hhgoa child-sweep`.
        self.child_sentences = max(child_sentences, 1)
        # Stride < child_sentences gives overlapping children, so a claim
        # spanning a sentence boundary still lands whole inside some window.
        self.child_stride = max(child_stride, 1)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.text.strip()
        if not text:
            return []

        sentences = split_sentences(text)
        if not sentences:
            return []

        size = self.child_sentences
        chunks: list[Chunk] = []
        idx = 0
        for start in range(0, len(sentences), self.child_stride):
            window = sentences[start : start + size]
            if not window:
                break
            child_text = " ".join(window).strip()
            if child_text:
                chunks.append(
                    Chunk(
                        id=f"{document.id}__c{idx}",
                        text=child_text,
                        # Parent identity, eval, citations, and dedup all key on this.
                        document_id=document.id,
                        language=document.language,
                        metadata={
                            **document.metadata,
                            "chunking": "parent_child",
                            "parent_id": document.id,
                            "sentence_start": start,
                            "sentence_end": start + len(window),
                            "parent_sentences": len(sentences),
                        },
                    )
                )
                idx += 1
            if start + size >= len(sentences):
                break
        return chunks
