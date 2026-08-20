"""ElevenLabs Scribe speech-to-text API."""

from __future__ import annotations

import httpx

from core.stt.base import BaseSTT, TranscriptionResult
from core.text import detect_language

_STT_LANGUAGES = {"hi", "gu"}
_DEFAULT_BASE_URL = "https://api.elevenlabs.io/v1"
_DEFAULT_MODEL = "scribe_v2"


class ElevenLabsSTT(BaseSTT):
    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = _DEFAULT_BASE_URL,
    ) -> None:
        if not api_key:
            raise ValueError("STT_API_KEY (or ELEVENLABS_API_KEY) is required for ElevenLabsSTT")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    def transcribe(
        self,
        audio: bytes,
        *,
        language: str | None = None,
        content_type: str = "audio/wav",
        filename: str = "audio.wav",
    ) -> TranscriptionResult:
        if language is not None and language not in _STT_LANGUAGES:
            raise ValueError(f"Unsupported STT language {language!r}. Supported: hi, gu")
        if not audio:
            raise ValueError("Audio payload is empty")

        headers = {"xi-api-key": self.api_key}
        # Omitting language_code lets Scribe detect the language itself, which
        # is what a speaker expects: they should not have to declare it first.
        data = {"model_id": self.model}
        if language is not None:
            data["language_code"] = language
        files = {"file": (filename, audio, content_type)}

        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/speech-to-text",
                headers=headers,
                data=data,
                files=files,
            )
            response.raise_for_status()
            payload = response.json()

        text = str(payload.get("text", "")).strip()
        detected = str(payload.get("language_code", "") or "")
        # The transcript settles it. Script detection cannot confuse Devanagari
        # with Gujarati, whereas the provider's own label occasionally does.
        resolved = detect_language(text) or (detected if detected in _STT_LANGUAGES else None)
        return TranscriptionResult(
            text=text,
            language=resolved or language or "hi",
            provider="elevenlabs",
            metadata={
                "model": self.model,
                "detected_language": detected,
                "language_probability": payload.get("language_probability"),
            },
        )
