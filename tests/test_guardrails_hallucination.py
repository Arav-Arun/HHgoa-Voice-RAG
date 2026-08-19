"""Tests for post-generation hallucination checking."""

from __future__ import annotations

from core.guardrails.hallucination import HallucinationChecker
from core.types import Chunk, ScoredChunk

CHECKER = HallucinationChecker(min_overlap=0.30)

SOURCE = [
    ScoredChunk(
        chunk=Chunk(
            id="c1",
            text="हैबर प्रक्रिया नाइट्रोजन और हाइड्रोजन से अमोनिया बनाती है।",
            document_id="d1",
            language="hi",
        ),
        score=0.82,
    )
]


def test_hallucination_blocks_ungrounded_answer() -> None:
    decision = CHECKER.check_answer(
        "हैबर प्रक्रिया क्या है?",
        "हैबर प्रक्रिया मंगल ग्रह पर पाई जाती है और सोने का उत्पादन करती है।",
        SOURCE,
        language="hi",
    )
    assert decision.blocked is True
    assert decision.reason == "low_context_overlap"
    assert decision.stage == "hallucination"


def test_hallucination_allows_grounded_answer() -> None:
    decision = CHECKER.check_answer(
        "हैबर प्रक्रिया क्या है?",
        "हैबर प्रक्रिया नाइट्रोजन और हाइड्रोजन से अमोनिया बनाती है।",
        SOURCE,
        language="hi",
    )
    assert decision.blocked is False
    assert decision.metadata["overlap"] >= 0.30
