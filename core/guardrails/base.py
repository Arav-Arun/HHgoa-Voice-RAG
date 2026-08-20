"""Guardrail interface, input filtering and grounding gates before the LLM."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from core.types import ScoredChunk


@dataclass
class GuardrailDecision:
    """Result of a guardrail check."""

    blocked: bool = False
    answer: str | None = None
    sources: list[ScoredChunk] | None = None
    reason: str | None = None
    stage: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseGuardrail(ABC):
    @abstractmethod
    def check_input(self, query: str, *, language: str = "hi") -> GuardrailDecision:
        """Validate query intent before retrieval."""

    @abstractmethod
    def check_grounding(
        self,
        query: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        """Validate retrieved context before calling the LLM."""

    def check_answer(
        self,
        query: str,
        answer: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        """Validate generated answer against retrieved context."""
        return GuardrailDecision(blocked=False, answer=answer, stage="hallucination")
