"""End-to-end RAG pipeline — wire retriever + guardrails + LLM here."""

from __future__ import annotations

from core.guardrails.base import BaseGuardrail, GuardrailDecision
from core.guardrails.stub import StubGuardrail
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
        guardrail: BaseGuardrail | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.default_language = default_language
        self.system_prompt = system_prompt
        self.guardrail = guardrail or StubGuardrail()

    def query(
        self,
        question: str,
        *,
        language: str | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        lang = language or self.default_language
        metadata: dict = {}

        input_decision = self.guardrail.check_input(question, language=lang)
        if input_decision.blocked:
            return self._blocked_response(
                question,
                input_decision,
                language=lang,
            )

        sources = self.retriever.retrieve(question, top_k=top_k)

        grounding_decision = self.guardrail.check_grounding(
            question,
            sources,
            language=lang,
        )
        if grounding_decision.blocked:
            return self._blocked_response(
                question,
                grounding_decision,
                language=lang,
            )

        sources = grounding_decision.sources or sources
        if grounding_decision.metadata:
            metadata["grounding"] = grounding_decision.metadata

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
            metadata=metadata,
        )

    @staticmethod
    def _blocked_response(
        question: str,
        decision: GuardrailDecision,
        *,
        language: str,
    ) -> RAGResponse:
        metadata = {
            "guardrail": {
                "blocked": True,
                "stage": decision.stage,
                "reason": decision.reason,
                **decision.metadata,
            }
        }
        return RAGResponse(
            query=question,
            answer=decision.answer or "",
            sources=decision.sources or [],
            language=language,
            metadata=metadata,
        )
