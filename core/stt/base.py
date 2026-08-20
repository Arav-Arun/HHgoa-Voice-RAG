"""Speech-to-text interface, swap providers without touching API routes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptionResult:
    text: str
    language: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSTT(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio: bytes,
        *,
        language: str = "hi",
        content_type: str = "audio/wav",
        filename: str = "audio.wav",
    ) -> TranscriptionResult:
        """Transcribe audio bytes to text in the target language."""
