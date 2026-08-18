"""Prompt templates — edit here to tune Hindi/Gujarati RAG behavior."""

from __future__ import annotations

from core.types import ScoredChunk

LANGUAGE_LABELS = {
    "hi": "Hindi (Devanagari)",
    "gu": "Gujarati",
}

SYSTEM_PROMPT = (
    "You are a helpful assistant for Hindi and Gujarati retrieval QA. "
    "Answer using only the provided context. If the context is insufficient, say so. "
    "Respond in the same language as the user's question when possible."
)


def build_rag_prompt(
    query: str,
    context_chunks: list[ScoredChunk],
    language: str = "hi",
) -> str:
    label = LANGUAGE_LABELS.get(language, language)
    if not context_chunks:
        context_block = "(no context retrieved)"
    else:
        parts = []
        for i, scored in enumerate(context_chunks, start=1):
            parts.append(f"[{i}] (score={scored.score:.3f})\n{scored.chunk.text}")
        context_block = "\n\n".join(parts)

    return (
        f"Language scope: {label}\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer concisely, citing passage numbers when relevant."
    )
