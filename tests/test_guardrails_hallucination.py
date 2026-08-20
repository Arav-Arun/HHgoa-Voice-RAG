"""Tests for post-generation hallucination checking."""

from __future__ import annotations

from core.guardrails.hallucination import HallucinationChecker
from core.types import Chunk, ScoredChunk

# Threshold sits between the two fixtures below: the ungrounded answer scores
# 0.31 (it shares only function/topic words: हैबर, प्रक्रिया, और, है) and the
# grounded one scores 1.00.
#
# This was 0.30 when the tokenizer folded the danda into the preceding word, so
# "है।" never matched a bare "है" and the ungrounded answer scored just under
# 0.30. core.text now strips sentence punctuation from tokens, which is correct
# and nudged that fixture to 0.3077: hence the wider margin here.
CHECKER = HallucinationChecker(min_overlap=0.50)

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
    assert decision.metadata["overlap"] >= 0.50
