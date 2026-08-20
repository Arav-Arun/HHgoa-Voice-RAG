"""Bounded in-process cache for the tier-2 LLM answer.

Tier 2 costs 2-6 seconds against a 9.6 ms fast path, so it is the only stage
where caching changes anything a user notices. Repeating a question, which a
demo does constantly, turns a 4.8 s wait into a lookup.

Deliberately in-process rather than Redis. The value is a small dict entry, the
lookup is nanoseconds, and a Redis round trip is 1-5 ms locally and 10-50 ms
across a region: for a pipeline whose whole budget is 9.6 ms, the network hop
would cost more than most of what it protects. Redis earns its place when
several instances need to share a cache, which is a scaling decision, not a
latency one.

Only the quality path is cached. The fast path is already cheaper than any
cache bookkeeping, and caching a guardrail decision would mean a policy change
could not take effect without a restart.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

DEFAULT_MAX_ENTRIES = 256


class AnswerCache:
    """LRU keyed on the exact question, language, and retrieved passages.

    The passage ids are part of the key on purpose: re-ingesting the corpus
    changes what a question retrieves, and an answer built from passages that
    are no longer top-ranked is stale rather than cached.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self.max_entries = max(int(max_entries), 0)
        self._entries: OrderedDict[tuple, Any] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.max_entries > 0

    @staticmethod
    def key(question: str, language: str, source_ids: list[str]) -> tuple:
        return (question.strip(), language, tuple(source_ids))

    def get(self, key: tuple) -> Any | None:
        if not self.enabled or key not in self._entries:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return self._entries[key]

    def put(self, key: tuple, value: Any) -> None:
        if not self.enabled:
            return
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}
