"""End-to-end RAG pipeline — wire retriever + LLM here."""

from __future__ import annotations

from core.llm.base import BaseLLM
from core.rag.prompts import SYSTEM_PROMPT
from core.retriever.base import BaseRetriever
from core.types import RAGResponse


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLLM,
        default_language: str = "hi",
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.default_language = default_language
        self.system_prompt = system_prompt

    def query(
        self,
        question: str,
        *,
        language: str | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        lang = language or self.default_language
        sources = self.retriever.retrieve(question, top_k=top_k)
        answer = self.llm.answer_with_context(
            question,
            sources,
            language=lang,
            system=self.system_prompt,
        )
        return RAGResponse(
            query=question,
            answer=answer,
            sources=sources,
            language=lang,
        )
