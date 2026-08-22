"""Extractive answerer, the deterministic, local fast path.

Returning whole retrieved passages is not an answer. A remote LLM is an
answer but costs a network round trip (measured: ~800 ms for STT alone), which
cannot fit the 200 ms budget. This sits in between: it selects the best-
supported sentence(s) from the retrieved context and returns them verbatim.

Two properties that matter:

* **Fast**, pure string work over a handful of candidate sentences, ~1 ms.
* **Grounded by construction**, every token of the answer is copied from the
  retrieved context, so this path cannot hallucinate. It can be wrong (it may
  select an unhelpful sentence) but it cannot invent.

It is also the terminal fallback of the harness: whatever else fails, there is
always a grounded answer to return.
"""

from __future__ import annotations

from core.chunking.sentences import split_sentences
from core.llm.base import BaseLLM
from core.text import off_script_terms, tokenize
from core.types import ScoredChunk

# A sentence needs at least this share of the query's informative mass to be
# considered responsive at all.
MIN_COVERAGE = 0.15


class ExtractiveAnswerer(BaseLLM):
    def __init__(
        self,
        *,
        max_sentences: int = 2,
        max_chars: int = 480,
        consider_chunks: int = 3,
        idf_lookup=None,
        embedder=None,
    ) -> None:
        self.max_sentences = max_sentences
        self.max_chars = max_chars
        self.consider_chunks = consider_chunks
        # Only consulted when the query carries a term the corpus script cannot
        # spell. See _dense_pick for why lexical scoring cannot cover that case.
        self.embedder = embedder
        # Callable term -> idf. Without one, every term weighs the same, which
        # lets ubiquitous words like "क्या"/"है" dominate sentence selection.
        self.idf_lookup = idf_lookup

    def generate(self, prompt: str, system: str | None = None) -> str:
        # No standalone generation: this answerer only ever works from context.
        return prompt

    def _weight(self, term: str) -> float:
        if self.idf_lookup is None:
            return 1.0
        # Floor at a small positive value so a term with no index entry still
        # counts a little rather than vanishing.
        return max(float(self.idf_lookup(term)), 0.05)

    def _dense_pick(self, query: str, context_chunks: list[ScoredChunk]) -> str:
        """Best sentence by embedding similarity, or "" if that is unavailable.

        Sentences are embedded in one batch so the escalation costs a single
        forward pass rather than one per candidate.
        """
        sentences: list[str] = []
        for scored_chunk in context_chunks[: self.consider_chunks]:
            sentences.extend(split_sentences(scored_chunk.chunk.text))
        if not sentences:
            return ""
        try:
            vectors = self.embedder.embed_texts([query, *sentences])
        except Exception:  # noqa: BLE001 - an embedder failure falls back to lexical
            return ""
        if len(vectors) != len(sentences) + 1:
            return ""
        query_vector, sentence_vectors = vectors[0], vectors[1:]
        best, best_score = "", float("-inf")
        for sentence, vector in zip(sentences, sentence_vectors, strict=True):
            score = sum(a * b for a, b in zip(query_vector, vector, strict=True))
            if score > best_score:
                best, best_score = sentence, score
        return best[: self.max_chars]

    def answer_with_context(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
        language: str = "hi",
        system: str | None = None,
    ) -> str:
        if not context_chunks:
            return ""

        # The answer is quoted verbatim, so quoting the parallel passage in the
        # other language answers a Hindi question in Gujarati. Measured on the
        # held-out set, that happened for 0.8% of queries. Filtering the
        # *retriever* by language was the obvious fix and turned out to be worth
        # nothing (+0.0019 hit@5, p=0.84), so the choice belongs here instead,
        # where it costs one list comprehension over five candidates.
        same_language = [c for c in context_chunks if c.chunk.language == language]
        context_chunks = same_language or context_chunks

        query_terms = set(tokenize(query))
        if not query_terms:
            return context_chunks[0].chunk.text[: self.max_chars]

        # "sodium" cannot match "सोडियम", so a query mixing scripts loses its
        # most informative term and picks a sentence on the leftovers. Measured:
        # asked for sodium, answered with calories from the same passage.
        # Escalate to the embedder, which is multilingual, and only then: this
        # costs an encode, where the lexical path costs a string scan.
        if self.embedder is not None and off_script_terms(query, language):
            dense = self._dense_pick(query, context_chunks)
            if dense:
                return dense

        total_weight = sum(self._weight(t) for t in query_terms) or 1.0

        scored: list[tuple[float, int, int, str]] = []
        for chunk_rank, scored_chunk in enumerate(context_chunks[: self.consider_chunks]):
            sentences = split_sentences(scored_chunk.chunk.text)
            for position, sentence in enumerate(sentences):
                sentence_terms = set(tokenize(sentence))
                matched = query_terms & sentence_terms
                if not matched:
                    continue
                coverage = sum(self._weight(t) for t in matched) / total_weight
                # Prefer earlier sentences and higher-ranked chunks on ties:
                # in MS MARCO passages the answer is usually stated up front.
                score = coverage - 0.02 * position - 0.05 * chunk_rank
                scored.append((score, chunk_rank, position, sentence))

        if not scored:
            # Nothing lexically responsive, fall back to the top passage's
            # opening sentence rather than returning nothing.
            top = split_sentences(context_chunks[0].chunk.text)
            return top[0][: self.max_chars] if top else ""

        scored.sort(key=lambda row: (-row[0], row[1], row[2]))
        if scored[0][0] < MIN_COVERAGE:
            best = scored[0][3]
            return best[: self.max_chars]

        # Keep the winners in reading order so multi-sentence answers stay coherent.
        chosen = sorted(scored[: self.max_sentences], key=lambda row: (row[1], row[2]))
        answer = " ".join(row[3] for row in chosen).strip()
        return answer[: self.max_chars]
