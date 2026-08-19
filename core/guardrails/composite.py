"""Composite guardrail — input intent filter then grounding gate."""

from __future__ import annotations

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.grounding import GroundingGate
from core.guardrails.hallucination import HallucinationChecker
from core.guardrails.input_intent import InputIntentFilter
from core.types import ScoredChunk


class CompositeGuardrail(BaseGuardrail):
    def __init__(
        self,
        input_filter: InputIntentFilter | None = None,
        grounding_gate: GroundingGate | None = None,
        hallucination_checker: HallucinationChecker | None = None,
    ) -> None:
        self.input_filter = input_filter or InputIntentFilter()
        self.grounding_gate = grounding_gate or GroundingGate()
        self.hallucination_checker = hallucination_checker or HallucinationChecker()

    def check_input(self, query: str, *, language: str = "hi") -> GuardrailDecision:
        return self.input_filter.check_input(query, language=language)

    def check_grounding(
        self,
        query: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        return self.grounding_gate.check_grounding(query, sources, language=language)

    def check_answer(
        self,
        query: str,
        answer: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        return self.hallucination_checker.check_answer(
            query,
            answer,
            sources,
            language=language,
        )
