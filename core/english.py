"""Original English text for indexed passages, for display alongside answers.

MS MARCO-XI is a translation of English MS MARCO, so every passage has a source
English text that ships with the dataset. Showing that beside a Hindi or
Gujarati answer is exact: it is what the passage was translated *from*, not a
machine translation of the answer back into English, so it cannot drift from
what the corpus says.

Loaded lazily and never consulted by retrieval or the guardrails. A missing file
simply means no English is shown; the pipeline is unaffected. Built by
``scripts/build_english_map.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

FILENAME = "passages_en.json"


class EnglishSources:
    """Maps ``document_id`` to the passage's original English text."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._mapping: dict[str, str] | None = None

    @property
    def mapping(self) -> dict[str, str]:
        if self._mapping is None:
            try:
                self._mapping = json.loads(self.path.read_text(encoding="utf-8"))
            except (AttributeError, OSError, json.JSONDecodeError):
                self._mapping = {}
        return self._mapping

    @property
    def available(self) -> bool:
        return bool(self.mapping)

    def get(self, document_id: str) -> str | None:
        return self.mapping.get(document_id)
