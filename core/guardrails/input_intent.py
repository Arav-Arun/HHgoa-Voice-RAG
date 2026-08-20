"""Input-intent filter, reject empty, abusive, or out-of-scope queries early."""

from __future__ import annotations

import re

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.messages import INPUT_REJECTED, UNSAFE_REJECTED, message_for
from core.types import ScoredChunk

_BLOCKED_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?(developer|admin|root)\s+mode", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system|hidden)\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+the\s+hidden\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+.*\s+prompt", re.IGNORECASE),
    re.compile(r"<\s*/?\s*script", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you\s+)?(know|learned)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(dan|jailbreak)", re.IGNORECASE),
    # Hindi injection phrasing
    re.compile(r"पिछले\s+निर्देश", re.IGNORECASE),
    re.compile(r"पिछले\s+सूचनाओं\s+को\s+नज़?रअंदाज़", re.IGNORECASE),
    re.compile(r"सिस्टम\s+प्रॉ?म्प्ट", re.IGNORECASE),
    re.compile(r"छुप.*प्रॉ?म्प्ट", re.IGNORECASE),
    re.compile(r"सभी\s+नियमों?\s+को\s+तोड़", re.IGNORECASE),
    re.compile(r"बिना\s+(किसी\s+)?नियम", re.IGNORECASE),
    # Gujarati injection phrasing
    re.compile(r"પાછલા\s+સૂચનાઓ", re.IGNORECASE),
    re.compile(r"સિસ્ટમ\s+પ્રોમ્પ્ટ", re.IGNORECASE),
    re.compile(r"છુપાયેલ\s+પ્રોમ્પ્ટ", re.IGNORECASE),
    re.compile(r"બધા\s+નિયમો\s+તોડ", re.IGNORECASE),
    re.compile(r"નિયમ\s+વગર", re.IGNORECASE),
)

# Unsafe-content patterns. These are refused **before retrieval**: a corpus
# lookup cannot make "how do I build a bomb" safe to answer, so confidence
# thresholds are the wrong tool. Kept separate from the injection patterns
# above so the two failure modes stay distinguishable in the metrics.
_UNSAFE_PATTERNS = (
    # English
    re.compile(r"\b(how\s+to\s+)?(make|build|create)\s+(a\s+)?(bomb|explosive|weapon)", re.IGNORECASE),
    re.compile(r"\bhack\s+(into\s+)?(someone|somebody|his|her|their|a)\b", re.IGNORECASE),
    re.compile(r"\b(steal|crack)\s+(a\s+)?(password|credentials|account)", re.IGNORECASE),
    re.compile(r"\b(counterfeit|forge)\s+(money|notes|currency|certificate|degree)", re.IGNORECASE),
    re.compile(r"\b(kill|poison|harm)\s+(someone|somebody|a\s+person)", re.IGNORECASE),
    re.compile(r"\b(commit\s+)?suicide\b|\bself[-\s]?harm\b", re.IGNORECASE),
    re.compile(r"\b(buy|purchase)\s+(illegal\s+)?(weapons?|guns?|drugs?)", re.IGNORECASE),
    re.compile(r"\b(evade|avoid)\s+(tax|taxes)\b|\btax\s+evasion\b", re.IGNORECASE),
    # Hindi
    re.compile(r"बम\s+(कैसे\s+)?बना", re.IGNORECASE),
    re.compile(r"(हैक|हैकिंग)\s*(कैसे|करूँ|करू|करना)", re.IGNORECASE),
    re.compile(r"नकली\s+(नोट|डिग्री|सर्टिफिकेट|प्रमाणपत्र)", re.IGNORECASE),
    re.compile(r"(ज़हर|जहर)\s+(कैसे\s+)?(दें|देना|दूँ)", re.IGNORECASE),
    re.compile(r"आत्महत्या", re.IGNORECASE),
    re.compile(r"(अवैध|गैरकानूनी)\s+(हथियार|शस्त्र|ड्रग)", re.IGNORECASE),
    re.compile(r"(पासवर्ड|आधार)\s+(\S+\s+)?(चुरा|चोरी|कैसे\s+पता|कैसे\s+जान)", re.IGNORECASE),
    re.compile(r"(चोरी|टैक्स\s+चोरी|कर\s+चोरी)\s*(कैसे)", re.IGNORECASE),
    re.compile(r"(मादक\s+पदार्थ|ड्रग्स)\s+.*बना", re.IGNORECASE),
    re.compile(r"(धमका|धमकी)", re.IGNORECASE),
    re.compile(r"जासूसी\s+कर", re.IGNORECASE),
    re.compile(r"ताला\s+(कैसे\s+)?तोड़", re.IGNORECASE),
    # Gujarati
    re.compile(r"બોમ્બ\s+(કેવી\s+રીતે\s+)?બના", re.IGNORECASE),
    re.compile(r"હેક\s*(કેવી|કરવું|કરવી)", re.IGNORECASE),
    re.compile(r"નકલી\s+(નોટ|ડિગ્રી|સર્ટિફિકેટ)", re.IGNORECASE),
    re.compile(r"ઝેર\s+(કેવી\s+રીતે\s+)?આપ", re.IGNORECASE),
    re.compile(r"આપઘાત|આત્મહત્યા", re.IGNORECASE),
    re.compile(r"ગેરકાયદે\s+(હથિયાર|શસ્ત્ર|ડ્રગ)", re.IGNORECASE),
    re.compile(r"(પાસવર્ડ|આધાર)\s+(\S+\s+)?(ચોર|કેવી\s+રીતે\s+જાણ)", re.IGNORECASE),
    re.compile(r"(ચોરી|કરચોરી)\s*(કેવી)", re.IGNORECASE),
    re.compile(r"ડ્રગ્સ\s+.*બના", re.IGNORECASE),
    re.compile(r"ધમકી", re.IGNORECASE),
    re.compile(r"જાસૂસી\s+કર", re.IGNORECASE),
    re.compile(r"તાળું\s+(કેવી\s+રીતે\s+)?તોડ", re.IGNORECASE),
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
        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(text):
                return GuardrailDecision(
                    blocked=True,
                    answer=message_for(language, UNSAFE_REJECTED),
                    reason="unsafe_content",
                    stage="input_intent",
                    metadata={"pattern": pattern.pattern},
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
