"""Post-generation faithfulness check, detect answers not supported by context.

Two independent signals:

* **Token overlap**, a bag-of-words share of the answer that appears in the
  retrieved context. Cheap, but a fluent model can recombine context words into
  a claim the context never made, so it is a floor, not a proof.
* **Numeric grounding**, every number appearing in the answer must appear in
  the context. Numbers are where ungrounded generation does real damage
  (prices, dates, dosages, rates), a wrong one is invisible to token overlap
  because the surrounding words all match, and the check costs one regex.
  Digits are script-normalized first, so a Gujarati "૫૦૦" in the passage
  grounds a "500" in the answer.
"""

from __future__ import annotations

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.messages import ABSTAIN_HALLUCINATION, message_for
from core.text import extract_numbers, token_set
from core.types import ScoredChunk


def _content_tokens(text: str) -> set[str]:
    # Shared tokenizer: matra-aware and danda-stripping, so "है।" and "है"
    # count as the same token instead of never matching.
    return token_set(text, min_length=2)


class HallucinationChecker(BaseGuardrail):
    def __init__(self, *, min_overlap: float = 0.20, check_numbers: bool = True) -> None:
        self.min_overlap = min_overlap
        self.check_numbers = check_numbers

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

        # A number the context never stated is ungrounded regardless of how
        # well the surrounding prose overlaps.
        ungrounded_numbers: list[str] = []
        if self.check_numbers:
            ungrounded_numbers = sorted(extract_numbers(text) - extract_numbers(context))
            if ungrounded_numbers:
                return GuardrailDecision(
                    blocked=True,
                    answer=message_for(language, ABSTAIN_HALLUCINATION),
                    sources=sources,
                    reason="ungrounded_numbers",
                    stage="hallucination",
                    metadata={
                        "ungrounded_numbers": ungrounded_numbers,
                        "overlap": round(overlap, 4),
                    },
                )

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
