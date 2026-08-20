"""Tools the generative model may call during the quality path.

This is what separates the quality path from a templated prompt: the model is
given the retriever as a callable and can decide the first context was
insufficient and search again, with a reformulated query, a different facet, or
a wider k. The loop is bounded so a confused model cannot spend the budget.

Every tool result is also recorded, so retrieved-but-unused passages still count
as valid citation targets.
"""

from __future__ import annotations

import json
from typing import Any

from core.retriever.base import BaseRetriever
from core.types import ScoredChunk

MAX_TOOL_ROUNDS = 2

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": (
                "Search the Hindi/Gujarati MS MARCO passage corpus. Use this when the "
                "context you were given does not contain the answer, for example to "
                "try a reformulated query or a more specific term."
            ),
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, in the same language as the question.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many passages to return (1-10).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_passage",
            "description": "Fetch the full text of one passage by its chunk id.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chunk_id"],
                "properties": {"chunk_id": {"type": "string"}},
            },
        },
    },
]


class ToolRunner:
    """Executes model tool calls against the real retriever."""

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        max_k: int = 10,
        language: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.max_k = max_k
        self.language = language
        # Everything the model has been shown, by chunk id, the citation
        # allow-list grows as the model searches.
        self.seen: dict[str, ScoredChunk] = {}
        self.calls: list[dict[str, Any]] = []

    def register(self, chunks: list[ScoredChunk]) -> None:
        for scored in chunks:
            self.seen.setdefault(scored.chunk.id, scored)

    def valid_ids(self) -> set[str]:
        return set(self.seen)

    def dispatch(self, name: str, arguments: str) -> str:
        """Run one tool call and return its JSON result string."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return json.dumps({"error": "arguments were not valid JSON"})

        self.calls.append({"name": name, "arguments": args})

        if name == "search_corpus":
            query = str(args.get("query", "")).strip()
            if not query:
                return json.dumps({"error": "query is required"})
            k = max(1, min(int(args.get("k", 5) or 5), self.max_k))
            results = self.retriever.retrieve(query, top_k=k, language=self.language)
            self.register(results)
            return json.dumps(
                {
                    "passages": [
                        {
                            "chunk_id": r.chunk.id,
                            "text": r.chunk.text,
                            "score": round(float(r.score), 4),
                        }
                        for r in results
                    ]
                },
                ensure_ascii=False,
            )

        if name == "get_passage":
            chunk_id = str(args.get("chunk_id", ""))
            scored = self.seen.get(chunk_id)
            if scored is None:
                return json.dumps({"error": f"unknown chunk_id {chunk_id!r}"})
            return json.dumps(
                {"chunk_id": chunk_id, "text": scored.chunk.text}, ensure_ascii=False
            )

        return json.dumps({"error": f"unknown tool {name!r}"})
