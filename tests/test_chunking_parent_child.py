"""Parent-child chunking: parent identity, overlap, and window sizing.

These pin the mechanism. The window size itself is chosen on the dev slice by
``uv run hhgoa child-sweep`` and recorded in ``data/eval/child-size-sweep.json``.
"""

from __future__ import annotations

from core.chunking.parent_child import ParentChildChunker
from core.types import Document

HI = "पहला वाक्य। दूसरा वाक्य। तीसरा वाक्य। चौथा वाक्य। पाँचवाँ वाक्य।"


def _doc(text: str, language: str) -> Document:
    return Document(id=f"{language}_1_p0", text=text, language=language)


def test_document_id_stays_the_parent():
    """Everything downstream resolves to the passage, not the child."""
    chunks = ParentChildChunker(child_sentences=2).chunk(_doc(HI, "hi"))
    assert len(chunks) > 1
    assert {c.document_id for c in chunks} == {"hi_1_p0"}
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.metadata["parent_id"] == "hi_1_p0" for c in chunks)


def test_children_overlap_so_a_claim_is_never_split():
    """Stride below the window size means consecutive children share a sentence."""
    chunks = ParentChildChunker(child_sentences=2, child_stride=1).chunk(_doc(HI, "hi"))
    spans = [(c.metadata["sentence_start"], c.metadata["sentence_end"]) for c in chunks]
    assert spans == [(0, 2), (1, 3), (2, 4), (3, 5)]


def test_a_wider_window_never_drops_the_tail():
    """Every sentence must appear in some child, whatever the window size."""
    for size in (1, 2, 3, 4, 9):
        chunker = ParentChildChunker(child_sentences=size)
        chunks = chunker.chunk(_doc(HI, "hi"))
        covered = set()
        for c in chunks:
            covered.update(range(c.metadata["sentence_start"], c.metadata["sentence_end"]))
        assert covered == set(range(chunks[0].metadata["parent_sentences"])), size


def test_degenerate_inputs():
    chunker = ParentChildChunker()
    assert chunker.chunk(_doc("   ", "hi")) == []
    # A single-sentence passage still yields exactly one child.
    single = chunker.chunk(_doc("अकेला वाक्य।", "hi"))
    assert len(single) == 1
    assert single[0].document_id == "hi_1_p0"
    # Sizes below 1 are clamped rather than producing empty windows.
    clamped = ParentChildChunker(child_sentences=0, child_stride=0)
    assert clamped.child_sentences == 1
    assert clamped.child_stride == 1
