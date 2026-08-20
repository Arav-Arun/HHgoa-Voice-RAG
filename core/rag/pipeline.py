"""End-to-end RAG pipeline.

Kept as a thin facade over :class:`core.harness.orchestrator.Orchestrator` so
that the CLI, the HTTP API, the eval harness, and the latency benchmark all
exercise exactly the same stage graph. If they diverged, the benchmark would
stop measuring the thing that ships.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.harness.contracts import QueryEnvelope
from core.text import detect_language
from core.types import RAGResponse

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    # core.harness.orchestrator imports core.rag.prompts, so importing it here
    # at runtime would close the loop core.rag -> harness -> core.rag.
    from core.harness.orchestrator import Orchestrator

# Fast path is local-only and budgeted; quality path may call a remote model.
DEFAULT_FAST_DEADLINE_MS = 200
DEFAULT_QUALITY_DEADLINE_MS = 30_000


class RAGPipeline:
    def __init__(
        self,
        orchestrator: Orchestrator,
        default_language: str = "hi",
    ) -> None:
        self.orchestrator = orchestrator
        self.default_language = default_language

    @property
    def retriever(self):
        return self.orchestrator.retriever

    @property
    def guardrail(self):
        return self.orchestrator.guardrail

    def query(
        self,
        question: str,
        *,
        language: str | None = None,
        top_k: int | None = None,
        mode: str = "fast",
        deadline_ms: int | None = None,
        trace_id: str = "",
    ) -> RAGResponse:
        if deadline_ms is None:
            deadline_ms = (
                DEFAULT_QUALITY_DEADLINE_MS if mode == "quality" else DEFAULT_FAST_DEADLINE_MS
            )
        # Devanagari and Gujarati are disjoint scripts, so the question states
        # its own language. An explicit argument still wins, for callers that
        # know better (eval fixtures carry a label).
        envelope = QueryEnvelope(
            text=question,
            language=language or detect_language(question) or self.default_language,
            top_k=top_k,
            mode=mode,  # type: ignore[arg-type]
            deadline_ms=deadline_ms,
            trace_id=trace_id,
        )
        return self.orchestrator.run(envelope)
