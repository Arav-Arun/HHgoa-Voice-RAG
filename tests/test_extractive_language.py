"""The extractive answer must be in the language that was asked.

MS MARCO-XI is parallel, so the same passage exists in Hindi and Gujarati and
retrieval can legitimately rank both. The answerer quotes a passage verbatim, so
picking the wrong copy answers the question in the wrong language.
"""

from __future__ import annotations

from core.llm.extractive import ExtractiveAnswerer
from core.types import Chunk, ScoredChunk

HI_TEXT = "बीमा समाधान स्वास्थ्य बीमा से संबंधित है। यह ग्राहकों की मदद करता है।"
GU_TEXT = "વીમા સમાધાન આરોગ્ય વીમા સાથે સંબંધિત છે. તે ગ્રાહકોને મદદ કરે છે."


def _source(text: str, language: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id=f"{language}_1__0", text=text, document_id=f"{language}_1", language=language),
        score=score,
        components={"dense_score": score},
    )


def test_answer_uses_the_queried_language_even_when_ranked_lower():
    # The Gujarati copy outranks the Hindi one; the Hindi question still gets
    # a Hindi answer.
    sources = [_source(GU_TEXT, "gu", 0.90), _source(HI_TEXT, "hi", 0.88)]
    answer = ExtractiveAnswerer().answer_with_context(
        "बीमा समाधान क्या है", sources, language="hi"
    )
    assert answer
    assert answer in HI_TEXT

    answer_gu = ExtractiveAnswerer().answer_with_context(
        "વીમા સમાધાન શું છે", sources, language="gu"
    )
    assert answer_gu in GU_TEXT


def test_falls_back_when_no_source_matches_the_language():
    """Better a grounded answer in the wrong language than no answer at all."""
    sources = [_source(GU_TEXT, "gu", 0.90)]
    answer = ExtractiveAnswerer().answer_with_context(
        "વીમા સમાધાન શું છે", sources, language="hi"
    )
    assert answer in GU_TEXT


def test_no_sources_yields_no_answer():
    assert ExtractiveAnswerer().answer_with_context("कुछ भी", [], language="hi") == ""
