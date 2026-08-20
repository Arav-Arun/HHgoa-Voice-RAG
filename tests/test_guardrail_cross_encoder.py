"""The cross-encoder feature: wiring, standardization, and graceful absence.

The model itself is not downloaded here. What matters for correctness is that
the score reaches the gate, that serving applies the same standardization the
calibration fitted, and that a checkout without a cross-encoder still works.
"""

from __future__ import annotations

from core.guardrails.confidence import (
    BASE_FEATURES,
    CROSS_FEATURE,
    ConfidenceGate,
    extract_features,
)
from core.types import Chunk, ScoredChunk


class _StubScorer:
    """Stands in for the cross-encoder; records what it was asked."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls: list[tuple[str, str]] = []

    def score(self, query: str, text: str) -> float:
        self.calls.append((query, text))
        return self.value


def _sources(*scores: float) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(id=f"c{i}", text=f"passage {i}", document_id=f"d{i}"),
            score=s,
            components={"dense_score": s},
        )
        for i, s in enumerate(scores)
    ]


def test_cross_score_is_added_only_when_a_scorer_is_given():
    src = _sources(0.9, 0.8)
    assert CROSS_FEATURE not in extract_features("q", src)
    scorer = _StubScorer(4.2)
    feats = extract_features("q", src, cross_scorer=scorer)
    assert feats[CROSS_FEATURE] == 4.2
    # Scored against the top passage only: verification, not reranking.
    assert scorer.calls == [("q", "passage 0")]


def test_empty_sources_keep_the_feature_shape():
    feats = extract_features("q", [], cross_scorer=_StubScorer(1.0))
    assert set(feats) == set(BASE_FEATURES) | {CROSS_FEATURE}
    assert feats[CROSS_FEATURE] == 0.0


def test_serving_applies_the_standardization_that_was_fitted():
    # One feature, coefficient 1, so confidence is the sigmoid of the z-score.
    gate = ConfidenceGate(
        coefficients=[1.0],
        intercept=0.0,
        threshold=0.5,
        feature_order=("top1_dense",),
        feature_mean=[0.5],
        feature_scale=[0.25],
    )
    # At the mean the z-score is 0, so confidence is exactly 0.5.
    assert abs(gate.confidence({"top1_dense": 0.5}) - 0.5) < 1e-9
    # Two standard deviations above the mean is sigmoid(2) = 0.881.
    assert abs(gate.confidence({"top1_dense": 1.0}) - 0.8808) < 1e-3
    assert abs(gate.confidence({"top1_dense": 0.0}) - 0.1192) < 1e-3


def test_a_model_without_standardization_still_loads():
    """Coefficients fitted before standardization must not be rescaled."""
    gate = ConfidenceGate(
        coefficients=[1.0], intercept=0.0, threshold=0.5, feature_order=("top1_dense",)
    )
    assert gate.feature_mean == [0.0]
    assert gate.feature_scale == [1.0]
    assert abs(gate.confidence({"top1_dense": 0.0}) - 0.5) < 1e-9


def test_scorer_is_ignored_when_the_fitted_model_does_not_use_it():
    """A 3-feature model must not pay for a forward pass it would discard."""
    scorer = _StubScorer(9.0)
    gate = ConfidenceGate(
        coefficients=[1.0, 0.0, 0.0],
        intercept=0.0,
        threshold=0.5,
        feature_order=BASE_FEATURES,
        cross_scorer=scorer,
    )
    assert gate.cross_scorer is None
    gate.check_grounding("q", _sources(0.9, 0.8))
    assert scorer.calls == []


def test_low_cross_score_can_block_when_it_dominates():
    gate = ConfidenceGate(
        coefficients=[0.0, 0.0, 0.0, 3.0],
        intercept=0.0,
        threshold=0.5,
        feature_order=(*BASE_FEATURES, CROSS_FEATURE),
        cross_scorer=_StubScorer(-5.0),
    )
    decision = gate.check_grounding("q", _sources(0.9, 0.8))
    assert decision.blocked
    assert decision.reason == "low_confidence"
