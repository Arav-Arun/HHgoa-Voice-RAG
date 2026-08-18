"""OpenAI-compatible chat completions client."""

from __future__ import annotations

import httpx

from core.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, system: str | None = None) -> str:
        if not self.api_key:
            return self._fallback(prompt)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

    @staticmethod
    def _fallback(prompt: str) -> str:
        """No API key — return retrieved context summary for local dev."""
        return (
            "[LLM_API_KEY not set — showing retrieved context only]\n\n"
            + prompt.split("Context:", maxsplit=1)[-1].strip()
        )
