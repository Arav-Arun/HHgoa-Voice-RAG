"""OpenAI-compatible Whisper transcription API."""

from __future__ import annotations

import httpx

from core.stt.base import BaseSTT, TranscriptionResult

_WHISPER_LANGUAGES = {"hi", "gu"}


class OpenAIWhisperSTT(BaseSTT):
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        base_url: str = "https://api.openai.com/v1",
    ) -> None:
        if not api_key:
            raise ValueError("STT_API_KEY or LLM_API_KEY is required for OpenAIWhisperSTT")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str = "hi",
        content_type: str = "audio/wav",
        filename: str = "audio.wav",
    ) -> TranscriptionResult:
        if language not in _WHISPER_LANGUAGES:
            raise ValueError(f"Unsupported STT language {language!r}. Supported: hi, gu")
        if not audio:
            raise ValueError("Audio payload is empty")

        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = {"model": self.model, "language": language}
        files = {"file": (filename, audio, content_type)}

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
            payload = response.json()

        text = str(payload.get("text", "")).strip()
        return TranscriptionResult(
            text=text,
            language=language,
            provider="openai",
            metadata={"model": self.model},
        )
