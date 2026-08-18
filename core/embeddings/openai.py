"""OpenAI-compatible embedding API — requires LLM_API_KEY + EMBEDDING_MODEL."""

from __future__ import annotations

import httpx

from core.embeddings.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimension: int | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("LLM_API_KEY is required for OpenAIEmbedder")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension = dimension or 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        payload: dict = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()["data"]
        ordered = sorted(data, key=lambda row: row["index"])
        return [row["embedding"] for row in ordered]
