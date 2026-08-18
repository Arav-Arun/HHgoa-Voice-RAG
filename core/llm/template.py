"""Template LLM — echoes context, useful for testing retrieval only."""

from __future__ import annotations

from core.llm.base import BaseLLM
from core.types import ScoredChunk


class TemplateLLM(BaseLLM):
    def generate(self, prompt: str, system: str | None = None) -> str:
        return prompt

    def answer_with_context(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
        language: str = "hi",
        system: str | None = None,
    ) -> str:
        if not context_chunks:
            return f"No relevant context found for: {query}"

        lines = [f"Query ({language}): {query}", "", "Top retrieved passages:"]
        for i, scored in enumerate(context_chunks, start=1):
            lines.append(f"{i}. [score={scored.score:.3f}] {scored.chunk.text}")
        return "\n".join(lines)
