"""Structured-output handling: parse, validate, and repair model JSON.

A model asked for JSON will occasionally return prose, fenced code, or JSON with
a missing field. Treating that as an unrecoverable error wastes a paid call;
treating it as valid corrupts the response. The harness does neither: it
attempts a tolerant parse, and if validation still fails it spends exactly one
repair round-trip before giving up and falling back.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from core.harness.contracts import AnswerPayload

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(ValueError):
    """Model output could not be coerced into the required schema."""


def extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON extraction from a model response."""
    text = (raw or "").strip()
    if not text:
        raise StructuredOutputError("empty model response")

    # Fenced block first, the most common wrapper when a model ignores
    # response_format.
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the outermost balanced-looking object.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise StructuredOutputError(f"no JSON object in response: {text[:120]!r}") from None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise StructuredOutputError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def parse_answer(raw: str) -> AnswerPayload:
    """Parse and validate a model response into an AnswerPayload."""
    try:
        return AnswerPayload.model_validate(extract_json(raw))
    except ValidationError as exc:
        raise StructuredOutputError(f"schema validation failed: {exc.error_count()} error(s)") from exc


def repair_prompt(raw: str, error: str) -> str:
    """Instruction for the single repair attempt."""
    return (
        "Your previous reply could not be parsed.\n"
        f"Error: {error}\n"
        f"Previous reply:\n{raw[:800]}\n\n"
        "Reply again with ONLY a JSON object matching the required schema. "
        "No prose, no markdown fences."
    )


def verify_citations(payload: AnswerPayload, valid_ids: set[str]) -> tuple[list[str], list[str]]:
    """Split cited ids into (valid, fabricated).

    A citation naming a passage that was never retrieved is evidence the model
    invented its support, which is a stronger hallucination signal than surface
    token overlap.
    """
    valid = [c for c in payload.citations if c in valid_ids]
    fabricated = [c for c in payload.citations if c not in valid_ids]
    return valid, fabricated
