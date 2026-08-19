"""Input-intent filter — reject empty, abusive, or out-of-scope queries early."""

from __future__ import annotations

import re

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.messages import INPUT_REJECTED, message_for
from core.types import ScoredChunk

_BLOCKED_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|admin|root)\s+mode", re.I),
    re.compile(r"reveal\s+(the\s+)?(system|hidden)\s+prompt", re.I),
    re.compile(r"reveal\s+the\s+hidden\s+prompt", re.I),
    re.compile(r"reveal\s+.*\s+prompt", re.I),
    re.compile(r"<\s*/?\s*script", re.I),
    re.compile(r"forget\s+(everything|all)\s+(you\s+)?(know|learned)", re.I),
    re.compile(r"act\s+as\s+(dan|jailbreak)", re.I),
    # Hindi injection phrasing
    re.compile(r"पिछले\s+निर्देश", re.I),
    re.compile(r"पिछले\s+सूचनाओं\s+को\s+नज़?रअंदाज़", re.I),
    re.compile(r"सिस्टम\s+प्रॉ?म्प्ट", re.I),
    re.compile(r"छुप.*प्रॉ?म्प्ट", re.I),
    re.compile(r"सभी\s+नियमों?\s+को\s+तोड़", re.I),
    # Gujarati injection phrasing
    re.compile(r"પાછલા\s+સૂચનાઓ", re.I),
    re.compile(r"સિસ્ટમ\s+પ્રોમ્પ્ટ", re.I),
    re.compile(r"છુપાયેલ\s+પ્રોમ્પ્ટ", re.I),
    re.compile(r"બધા\s+નિયમો\s+તોડ", re.I),
)


class InputIntentFilter(BaseGuardrail):
    def __init__(
        self,
        *,
        min_query_length: int = 3,
        supported_languages: tuple[str, ...] = ("hi", "gu"),
    ) -> None:
        self.min_query_length = min_query_length
        self.supported_languages = supported_languages

    def check_input(self, query: str, *, language: str = "hi") -> GuardrailDecision:
        text = query.strip()
        if not text:
            return GuardrailDecision(
                blocked=True,
                answer=message_for(language, INPUT_REJECTED),
                reason="empty_query",
                stage="input_intent",
            )
        if len(text) < self.min_query_length:
            return GuardrailDecision(
                blocked=True,
                answer=message_for(language, INPUT_REJECTED),
                reason="query_too_short",
                stage="input_intent",
                metadata={"length": len(text), "min_length": self.min_query_length},
            )
        if language not in self.supported_languages:
            return GuardrailDecision(
                blocked=True,
                answer=message_for(language, INPUT_REJECTED),
                reason="unsupported_language",
                stage="input_intent",
                metadata={"language": language},
            )
        for pattern in _BLOCKED_PATTERNS:
            if pattern.search(text):
                return GuardrailDecision(
                    blocked=True,
                    answer=message_for(language, INPUT_REJECTED),
                    reason="blocked_intent",
                    stage="input_intent",
                    metadata={"pattern": pattern.pattern},
                )
        return GuardrailDecision(blocked=False, stage="input_intent")

    def check_grounding(
        self,
        query: str,
        sources: list[ScoredChunk],
        *,
        language: str = "hi",
    ) -> GuardrailDecision:
        return GuardrailDecision(blocked=False, sources=sources, stage="grounding")
