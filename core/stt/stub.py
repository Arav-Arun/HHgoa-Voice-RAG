"""Stub STT — used when no provider is configured."""

from __future__ import annotations

from core.stt.base import BaseSTT, TranscriptionResult


class StubSTT(BaseSTT):
    def transcribe(
        self,
        audio: bytes,
        *,
        language: str = "hi",
        content_type: str = "audio/wav",
        filename: str = "audio.wav",
    ) -> TranscriptionResult:
        raise RuntimeError(
            "STT is not configured. Set STT_PROVIDER=elevenlabs and STT_API_KEY "
            "(or ELEVENLABS_API_KEY) in .env, then restart the server."
        )
