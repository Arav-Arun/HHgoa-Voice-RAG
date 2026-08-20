"""OpenAI-compatible chat client with tool-calling and structured output.

Deliberately a thin transport layer: it performs exactly one HTTP request per
call and raises on failure. It does **not** retry, back off, or fall back -
those are budget decisions and belong to the harness, which is the only layer
that knows how much of the deadline is left.

Works unmodified against any OpenAI-compatible endpoint (OpenAI, Groq,
Together, a local vLLM) since they share the /chat/completions contract; only
``base_url`` and ``model`` change.
"""

from __future__ import annotations

from typing import Any

import httpx


class ChatClient:
    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: float = 20.0,
        provider: str = "openai",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.provider = provider

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_schema: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """One chat completion. Returns the raw assistant message object."""
        if not self.api_key:
            raise RuntimeError("ChatClient has no API key configured")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}

        with httpx.Client(timeout=timeout or self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            # raise_for_status gives the harness an httpx.HTTPStatusError whose
            # status code its retry policy already knows how to classify.
            response.raise_for_status()
            body = response.json()

        return body["choices"][0]["message"]
