"""Multi-feature grounding gate.

A single cosine threshold cannot separate answerable from unanswerable queries
on this corpus: measured on the held-out eval slice, answerable top-1 cosines
average 0.884 and unanswerable ones average 0.864. The distributions genuinely
overlap, so any threshold trades one error for the other almost one-for-one.
The 0.86 default refuses 16.2% of *answerable* questions to catch 41.7% of
unanswerable ones, and the best swept threshold that keeps answerable recall
above 95% catches only 6.3%.

Lexical overlap between the query and the top passage separates far better
(0.617 vs 0.333). Combining it with the cosine and the top-1 margin in a small
logistic regression, fitted on a calibration half and reported on a held-out
half, cuts the false-abstain rate from 16.2% to 4.5%. It gives up some abstain
recall to do it (0.417 to 0.333), which is the intended trade: refusing a real
question is the more damaging error here. Balanced accuracy rises 0.627 -> 0.644.

A fourth feature, ``cross_score``, comes from a cross-encoder that reads the
query and the top passage together (see :mod:`core.guardrails.cross_encoder`).
It is the only feature that separates strongly: +2.54 answerable against -3.57
should-abstain, where the dense cosine separates by 0.02. Adding it lifts
abstain recall from 0.306 to 0.379 at the same false-abstain rate.

``margin`` survives as a feature but carries almost no weight. It mattered more
when retrieval ranked chunks, because sibling chunks of one passage produced
near-identical top scores; ranking passages removed that artifact.

Metrics are averaged over 30 random half-splits: one split leaves ~48 abstain
examples and its balanced accuracy swings by +/-0.03 between seeds.

Coefficients live in ``data/eval/guardrail-model.json``, written by
``uv run hhgoa guardrail-calibrate``. If that file is absent the gate degrades to the
plain cosine threshold, so the system still runs on a fresh checkout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.messages import (
    ABSTAIN_LOW_CONFIDENCE,
    ABSTAIN_NO_CONTEXT,
    message_for,
)
from core.text import token_set
from core.types import ScoredChunk

DEFAULT_MODEL_PATH = Path("data/eval/guardrail-model.json")
BASE_FEATURES = ("top1_dense", "margin", "lexical_overlap")
# `cross_score` is appended only when a cross-encoder is configured, so a fitted
# model records which set it was trained on and a fresh checkout still runs.
CROSS_FEATURE = "cross_score"
FEATURE_ORDER = BASE_FEATURES


class _LogisticModel:
    """A fitted logistic gate: standardize, dot, sigmoid."""

    def __init__(
        self,
        features: tuple[str, ...],
        coefficients: list[float],
        intercept: float,
        threshold: float,
        mean: list[float] | None = None,
        scale: list[float] | None = None,
    ) -> None:
        self.features = features
        self.coefficients = coefficients
        self.intercept = intercept
        self.threshold = threshold
        self.mean = mean or [0.0] * len(features)
        self.scale = scale or [1.0] * len(features)

    @classmethod
    def from_dict(cls, data: dict) -> _LogisticModel:
        return cls(
            features=tuple(data.get("features", FEATURE_ORDER)),
            coefficients=[float(c) for c in data["coefficients"]],
            intercept=float(data["intercept"]),
            threshold=float(data["threshold"]),
            mean=data.get("feature_mean"),
            scale=data.get("feature_scale"),
        )

    def confidence(self, features: dict[str, float]) -> float:
        z = self.intercept
        for coef, name, mu, sd in zip(
            self.coefficients, self.features, self.mean, self.scale, strict=False
        ):
            z += coef * ((features.get(name, 0.0) - mu) / (sd or 1.0))
        return 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))


def extract_features(
    query: str,
    sources: list[ScoredChunk],
    *,
    cross_scorer=None,
) -> dict[str, float]:
    """The signals the gate is calibrated on.

    Shared by the runtime gate and the calibration script so the features can
    never drift apart between fitting and serving. Passing ``cross_scorer``
    adds the cross-encoder relevance score for the top passage.
    """
    if not sources:
        base = {"top1_dense": 0.0, "margin": 0.0, "lexical_overlap": 0.0}
        return {**base, CROSS_FEATURE: 0.0} if cross_scorer else base

    dense = [s.dense_score for s in sources]
    top1 = max(dense)
    rest = dense[1:] or [top1]
    query_tokens = token_set(query)
    top_tokens = token_set(sources[0].chunk.text)
    overlap = len(query_tokens & top_tokens) / len(query_tokens) if query_tokens else 0.0
    features = {
        "top1_dense": top1,
        "margin": top1 - (sum(rest) / len(rest)),
        "lexical_overlap": overlap,
    }
    if cross_scorer is not None:
        features[CROSS_FEATURE] = cross_scorer.score(query, sources[0].chunk.text)
    return features


class ConfidenceGate(BaseGuardrail):
    def __init__(
        self,
        *,
        coefficients: list[float] | None = None,
        intercept: float = 0.0,
        threshold: float = 0.5,
        min_score: float = 0.86,
        feature_order: tuple[str, ...] = FEATURE_ORDER,
        feature_mean: list[float] | None = None,
        feature_scale: list[float] | None = None,
        cascade: _LogisticModel | None = None,
        cascade_band: tuple[float, float] = (0.0, 1.0),
        cross_scorer=None,
    ) -> None:
        self.coefficients = coefficients
        self.intercept = intercept
        self.threshold = threshold
        # Used only when no fitted model is available.
        self.min_score = min_score
        self.feature_order = feature_order
        # Standardization fitted alongside the coefficients; identity when the
        # model predates it.
        self.feature_mean = feature_mean or [0.0] * len(feature_order)
        self.feature_scale = feature_scale or [1.0] * len(feature_order)
        # Only consulted when the fitted model actually uses the feature, so a
        # 3-feature model never pays for a forward pass it will ignore.
        self.cross_scorer = cross_scorer if CROSS_FEATURE in feature_order else None
        # Cheap pre-gate. When its verdict is already decisive the cross-encoder
        # is skipped entirely; measured, this halves how often it runs and
        # changes none of the gate's metrics.
        self.cascade = cascade
        self.cascade_band = cascade_band

    @classmethod
    def from_file(
        cls,
        path: Path = DEFAULT_MODEL_PATH,
        *,
        min_score: float = 0.86,
        cross_scorer=None,
    ) -> ConfidenceGate:
        if not path.exists():
            return cls(min_score=min_score)
        data = json.loads(path.read_text(encoding="utf-8"))
        cascade_data = data.get("cascade")
        return cls(
            coefficients=[float(c) for c in data["coefficients"]],
            intercept=float(data["intercept"]),
            threshold=float(data["threshold"]),
            min_score=min_score,
            feature_order=tuple(data.get("features", FEATURE_ORDER)),
            feature_mean=data.get("feature_mean"),
            feature_scale=data.get("feature_scale"),
            cascade=_LogisticModel.from_dict(cascade_data) if cascade_data else None,
            cascade_band=tuple(cascade_data["band"]) if cascade_data else (0.0, 1.0),
            cross_scorer=cross_scorer,
        )

    @property
    def fitted(self) -> bool:
        return self.coefficients is not None

    def confidence(self, features: dict[str, float]) -> float:
        """P(answerable). Falls back to a pass/fail on cosine when unfitted."""
        if self.coefficients is None:
            return 1.0 if features["top1_dense"] >= self.min_score else 0.0
        return _LogisticModel(
            self.feature_order,
            self.coefficients,
            self.intercept,
            self.threshold,
            self.feature_mean,
            self.feature_scale,
        ).confidence(features)

    def score(self, query: str, sources: list[ScoredChunk]) -> tuple[float, float, dict]:
        """Confidence, the threshold it must clear, and the features used.

        Runs the cheap features first. The cross-encoder is only consulted when
        their verdict is inside the undecided band, which is what keeps the
        median query from paying for a second transformer pass.
        """
        features = extract_features(query, sources)
        if self.cascade is None or self.cross_scorer is None:
            if self.cross_scorer is not None:
                features[CROSS_FEATURE] = self.cross_scorer.score(query, sources[0].chunk.text)
            return self.confidence(features), self.threshold, features

        cheap = self.cascade.confidence(features)
        low, high = self.cascade_band
        if not (low < cheap < high):
            features["cascade"] = 1.0
            return cheap, self.cascade.threshold, features

        features[CROSS_FEATURE] = self.cross_scorer.score(query, sources[0].chunk.text)
        return self.confidence(features), self.threshold, features

    def check_input(self, query: str, *, language: str = "hi") -> GuardrailDecision:
        return GuardrailDecision(blocked=False, stage="input_intent")

    def check_grounding(
        self,
        query: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        if not sources:
            return GuardrailDecision(
                blocked=True,
                answer=message_for(language, ABSTAIN_NO_CONTEXT),
                sources=[],
                reason="no_context",
                stage="grounding",
            )

        confidence, threshold, features = self.score(query, sources)
        detail = {
            "confidence": round(confidence, 4),
            "threshold": threshold if self.fitted else self.min_score,
            "model": "logistic" if self.fitted else "threshold",
            **{k: round(v, 4) for k, v in features.items()},
        }

        if confidence < threshold:
            return GuardrailDecision(
                blocked=True,
                answer=message_for(language, ABSTAIN_LOW_CONFIDENCE),
                sources=sources,
                reason="low_confidence",
                stage="grounding",
                metadata=detail,
            )

        return GuardrailDecision(blocked=False, sources=sources, stage="grounding", metadata=detail)
