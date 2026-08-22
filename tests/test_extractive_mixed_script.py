"""A query mixing scripts must not lose its most informative term.

"हरी मिर्च में sodium कितना है" tokenizes `sodium` in Latin while the corpus
spells it `सोडियम`. Lexical scoring cannot match the two, so it answered from
whatever else overlapped: asked for sodium, returned the calories sentence from
the same passage. The answerer escalates to the embedder for exactly this case.
"""

from __future__ import annotations

from core.llm.extractive import ExtractiveAnswerer
from core.text import off_script_terms
from core.types import Chunk, ScoredChunk

SODIUM = "1 कप कच्चे हरे बेल मिर्च में 4 मिलीग्राम सोडियम होता है।"
CALORIES = "लगभग 150 ग्राम के एक कप कटी हुई हरी मिर्च में 30 कैलोरी होती हैं।"
PASSAGE = f"{CALORIES} {SODIUM}"


class StubEmbedder:
    """Scores by term overlap against a fixed bilingual pairing.

    Standing in for e5 keeps the test deterministic and offline. The property
    under test is that the answerer consults an embedder at all, not how well
    that particular model bridges scripts.
    """

    BRIDGE = {"sodium": "सोडियम", "vitamin": "विटामिन"}

    @property
    def dimension(self) -> int:
        return 1

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        query, *rest = texts
        wanted = [self.BRIDGE[t] for t in query.lower().split() if t in self.BRIDGE]
        # The query vector is 1.0; each sentence scores 1 when it carries the
        # bridged term, so the dot product ranks the responsive sentence first.
        return [[1.0]] + [[1.0 if any(w in s for w in wanted) else 0.0] for s in rest]


def _source() -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id="hi_1__0", text=PASSAGE, document_id="hi_1", language="hi"),
        score=0.9,
        components={"dense_score": 0.9},
    )


def test_off_script_terms_names_the_unmatchable_word():
    assert off_script_terms("हरी मिर्च में sodium कितना है?", "hi") == {"sodium"}
    # Written in the corpus script there is nothing to escalate for.
    assert off_script_terms("हरी मिर्च में सोडियम कितना है?", "hi") == set()
    # Digits carry no script and match across both, so they are not a trigger.
    assert off_script_terms("100 ग्राम में कितना", "hi") == set()


def test_latin_term_in_hindi_query_reaches_the_matching_sentence():
    answerer = ExtractiveAnswerer(embedder=StubEmbedder())
    answer = answerer.answer_with_context("हरी मिर्च में sodium कितना है", [_source()])
    assert answer == SODIUM


def test_without_an_embedder_the_lexical_path_still_answers():
    # The escalation is optional: a build with no embedder must degrade to
    # lexical selection rather than fail.
    answer = ExtractiveAnswerer().answer_with_context(
        "हरी मिर्च में sodium कितना है", [_source()]
    )
    assert answer in PASSAGE


def test_same_script_query_does_not_pay_for_an_encode():
    class Explodes(StubEmbedder):
        def embed_texts(self, texts):  # noqa: ARG002 - must never be reached
            raise AssertionError("embedder consulted for a same-script query")

    answerer = ExtractiveAnswerer(embedder=Explodes())
    answer = answerer.answer_with_context("सोडियम कितना है", [_source()])
    assert answer in PASSAGE
