"""Cross-encoder relevance scoring for the grounding gate.

The other three gate features are bag-of-words or a bi-encoder cosine, and none
of them can see that a passage is about the *wrong* variant of the right thing.
Measured case: "how much sodium in **red** pepper" against a passage stating the
sodium content of **green** pepper scores 0.881 dense and 0.571 lexical overlap,
both squarely inside the answerable range, because every token matches. The
discriminating word is one adjective and it carries no more weight than "is".

A cross-encoder reads the query and the passage *together*, so the contradiction
is visible to it. On that same pair it scores 2.19, against 7.15 for the same
question asked about green pepper.

Loading is lazy and the model is cached per process, because the gate is
constructed during calibration too and a 60s download should happen once.
"""

from __future__ import annotations

from typing import ClassVar

DEFAULT_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# Passages are ~300 characters; 256 tokens covers query + passage with room to
# spare, and shorter inputs keep the check inside the latency budget.
MAX_LENGTH = 256
# Only the top passage is scored. This is a verification step, not a reranker:
# one pair costs ~10ms, five cost ~20ms, and the answer is drawn from the top
# passage anyway.
MAX_TEXT_CHARS = 600


class CrossEncoderScorer:
    """Scores how well a passage answers a query. Higher is better."""

    _models: ClassVar[dict[str, object]] = {}

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name

    @property
    def model(self):
        if self.model_name not in self._models:
            from sentence_transformers import CrossEncoder

            self._models[self.model_name] = CrossEncoder(self.model_name, max_length=MAX_LENGTH)
        return self._models[self.model_name]

    def score(self, query: str, text: str) -> float:
        if not query or not text:
            return 0.0
        return float(self.model.predict([(query, text[:MAX_TEXT_CHARS])])[0])

    def score_many(self, query: str, texts: list[str]) -> list[float]:
        if not query or not texts:
            return []
        pairs = [(query, t[:MAX_TEXT_CHARS]) for t in texts]
        return [float(s) for s in self.model.predict(pairs)]

    def warm(self) -> None:
        """Force the download and first forward pass out of the request path."""
        self.score("warmup", "warmup passage")
