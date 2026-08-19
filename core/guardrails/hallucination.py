"""Post-generation faithfulness check — detect answers not supported by context."""

from __future__ import annotations

import re

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.messages import ABSTAIN_HALLUCINATION, message_for
from core.types import ScoredChunk

_TOKEN_RE = re.compile(r"[\w\u0900-\u097F\u0A80-\u0AFF]+", re.UNICODE)


def _content_tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) >= 2}


class HallucinationChecker(BaseGuardrail):
    def __init__(self, *, min_overlap: float = 0.20) -> None:
        self.min_overlap = min_overlap

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

    def check_answer(
        self,
        query: str,
        answer: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        text = answer.strip()
        if not text or not sources:
            return GuardrailDecision(blocked=False, answer=answer, stage="hallucination")

        context = " ".join(scored.chunk.text for scored in sources)
        answer_tokens = _content_tokens(text)
        if not answer_tokens:
            return GuardrailDecision(blocked=False, answer=answer, stage="hallucination")

        context_tokens = _content_tokens(context)
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)

        if overlap < self.min_overlap:
            return GuardrailDecision(
                blocked=True,
                answer=message_for(language, ABSTAIN_HALLUCINATION),
                sources=sources,
                reason="low_context_overlap",
                stage="hallucination",
                metadata={
                    "overlap": round(overlap, 4),
                    "min_overlap": self.min_overlap,
                    "answer_token_count": len(answer_tokens),
                    "matched_token_count": len(answer_tokens & context_tokens),
                },
            )

        return GuardrailDecision(
            blocked=False,
            answer=answer,
            sources=sources,
            stage="hallucination",
            metadata={"overlap": round(overlap, 4)},
        )
