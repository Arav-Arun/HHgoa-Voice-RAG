"""No-op guardrail, passthrough for tests or disabled guardrails."""

from __future__ import annotations

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.types import ScoredChunk


class StubGuardrail(BaseGuardrail):
    def check_input(self, query: str, *, language: str = "hi") -> GuardrailDecision:
        return GuardrailDecision(blocked=False, stage="input_intent")

    def check_grounding(
        self,
        query: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        return GuardrailDecision(blocked=False, sources=sources, stage="grounding")
