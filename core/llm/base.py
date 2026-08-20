"""LLM interface for answer generation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import ScoredChunk


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text from a user prompt."""

    def answer_with_context(
        self,
        query: str,
        context_chunks: list[ScoredChunk],
        language: str = "hi",
        system: str | None = None,
    ) -> str:
        """Default RAG answer path, override for custom prompting."""
        from core.rag.prompts import build_rag_prompt

        prompt = build_rag_prompt(query, context_chunks, language=language)
        return self.generate(prompt, system=system)
