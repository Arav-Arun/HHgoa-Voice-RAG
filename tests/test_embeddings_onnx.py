"""The ONNX encoder must produce torch's vectors, or not be used at all.

The corpus was encoded with torch. An encoder that disagrees would not fail,
it would quietly rank queries against different vectors and lose recall, which
is the failure mode worth a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.embeddings.onnx import OnnxEmbedder, export_path
from core.embeddings.sentence_transformers import SentenceTransformerEmbedder

MODEL = "intfloat/multilingual-e5-small"
EXPORT = export_path(MODEL)
HAS_EXPORT = (EXPORT / "model.onnx").exists()


def test_missing_export_falls_back_to_torch(tmp_path: Path):
    # A deployment without the export must keep working, just at torch speed.
    embedder = OnnxEmbedder(MODEL, cache_dir=tmp_path)
    assert embedder._session() is None
    assert isinstance(embedder.torch_embedder, SentenceTransformerEmbedder)


def test_export_path_is_stable_and_contained():
    # Slashes in a model name must not escape the cache directory.
    path = export_path("intfloat/multilingual-e5-small", Path("data/onnx"))
    assert path.parent == Path("data/onnx")
    assert "/" not in path.name
    assert path == export_path("intfloat/multilingual-e5-small", Path("data/onnx"))


@pytest.mark.skipif(not HAS_EXPORT, reason="run scripts/export_onnx.py first")
def test_onnx_vectors_match_torch():
    import numpy as np

    onnx = OnnxEmbedder(MODEL, query_prefix="query: ", passage_prefix="passage: ")
    torch_side = onnx.torch_embedder
    assert onnx._session() is not None, "export present but session failed to build"

    for probe in (
        "लाल मिर्च में कौन सा विटामिन होता है",
        "સૌથી વધુ રોકડ પુરસ્કાર ક્રેડિટ કાર્ડ્સ",
    ):
        a = np.array(onnx.embed_query(probe))
        b = np.array(torch_side.embed_query(probe))
        # Both are unit vectors, so the dot product is the cosine.
        assert float(a @ b) > 0.999999
