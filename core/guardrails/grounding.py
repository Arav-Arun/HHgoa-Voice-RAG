"""Grounding / abstain gate, require sufficient retrieval confidence before LLM.

Calibration note: this gate thresholds the **dense cosine similarity**, read via
``ScoredChunk.dense_score``, never ``ScoredChunk.score``. Under hybrid retrieval
``score`` is a Reciprocal Rank Fusion value in the ~0.01-0.05 range, which has no
absolute meaning and is not comparable to the cosine-calibrated threshold, using
it would abstain on every query.
"""

from __future__ import annotations

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.messages import (
    ABSTAIN_LOW_CONFIDENCE,
    ABSTAIN_NO_CONTEXT,
    message_for,
)
from core.types import ScoredChunk


class GroundingGate(BaseGuardrail):
    def __init__(self, *, min_score: float = 0.30) -> None:
        self.min_score = min_score

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

        filtered = [scored for scored in sources if scored.dense_score >= self.min_score]
        best_score = max(scored.dense_score for scored in sources)

        if not filtered:
            return GuardrailDecision(
                blocked=True,
                answer=message_for(language, ABSTAIN_LOW_CONFIDENCE),
                sources=sources,
                reason="low_confidence",
                stage="grounding",
                metadata={
                    "best_score": best_score,
                    "min_score": self.min_score,
                    "source_count": len(sources),
                },
            )

        return GuardrailDecision(
            blocked=False,
            sources=filtered,
            stage="grounding",
            metadata={
                "best_score": best_score,
                "min_score": self.min_score,
                "kept_sources": len(filtered),
                "dropped_sources": len(sources) - len(filtered),
            },
        )
