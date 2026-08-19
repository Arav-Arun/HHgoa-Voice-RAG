"""Shared sentence splitting for Indic + Latin scripts."""

from __future__ import annotations

import re

# Devanagari/Gujarati danda, Latin punctuation, paragraph breaks.
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[।॥.!??\u0964\u0965])\s+|\n\n+",
    flags=re.UNICODE,
)


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_BOUNDARY.split(text)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences or [text]
