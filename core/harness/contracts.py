"""Typed contracts for the orchestration harness.

Every boundary in the pipeline, what goes in, what each stage returns, what the
model is required to emit, is a declared Pydantic model rather than a loose
dict. That is what makes the difference between "a prompt call" and a harness:
malformed model output is a *validation error the harness can act on*, not a
string that silently flows downstream.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Mode = Literal["fast", "quality"]
StageStatus = Literal["ok", "blocked", "failed", "skipped", "fallback", "timeout"]


class QueryEnvelope(BaseModel):
    """Validated input to the pipeline."""

    text: str
    language: str = Field(default="hi", pattern="^(hi|gu)$")
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Mode = "fast"
    # Wall-clock budget for the whole run. The fast path is designed to finish
    # inside 200 ms; the quality path is allowed to overrun and is measured
    # separately.
    deadline_ms: int = Field(default=200, ge=1, le=120_000)
    trace_id: str = ""

    @field_validator("text")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()


class AnswerPayload(BaseModel):
    """The structured output the generative model is required to produce.

    ``citations`` are chunk ids and are verified against the retrieved set, a
    model that cites a passage it was never shown has hallucinated its evidence,
    which is a stronger signal than token overlap alone.
    """

    answer: str
    citations: list[str] = Field(default_factory=list)
    sufficient: bool = True
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("answer")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer must not be empty")
        return value


# JSON Schema handed to the model; kept next to the model it must satisfy.
ANSWER_JSON_SCHEMA: dict[str, Any] = {
    "name": "grounded_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["answer", "citations", "sufficient", "confidence"],
        "properties": {
            "answer": {
                "type": "string",
                "description": "Answer in the user's language, using only the provided context.",
            },
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ids of the context passages that support the answer.",
            },
            "sufficient": {
                "type": "boolean",
                "description": "False if the context does not actually answer the question.",
            },
            "confidence": {"type": "number", "description": "0.0 to 1.0."},
        },
    },
}


class StageResult(BaseModel):
    name: str
    status: StageStatus
    ms: float
    attempts: int = 1
    error: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """Per-run execution record. Backs both debugging and the latency report."""

    trace_id: str
    path: str = "fast"
    mode: Mode = "fast"
    stages: list[StageResult] = Field(default_factory=list)
    total_ms: float = 0.0

    def add(self, result: StageResult) -> StageResult:
        self.stages.append(result)
        return result

    def timings(self) -> dict[str, float]:
        """Per-stage milliseconds, for the latency breakdown."""
        return {stage.name: round(stage.ms, 3) for stage in self.stages}

    def failed_stages(self) -> list[str]:
        return [s.name for s in self.stages if s.status in {"failed", "timeout"}]
