"""Tier-2 answer caching: what it keys on, and what it refuses to keep."""

from __future__ import annotations

from core.harness.cache import AnswerCache


def test_disabled_cache_never_stores():
    cache = AnswerCache(0)
    assert not cache.enabled
    key = cache.key("q", "hi", ["a"])
    cache.put(key, "value")
    assert cache.get(key) is None


def test_hit_and_miss():
    cache = AnswerCache(4)
    key = cache.key("बीमा समाधान क्या है", "hi", ["c1", "c2"])
    assert cache.get(key) is None
    cache.put(key, ("answer", ["c1"], {"path": "quality"}))
    assert cache.get(key)[0] == "answer"
    assert cache.stats()["hits"] == 1


def test_key_includes_language_and_retrieved_passages():
    """Re-ingesting changes what a question retrieves, so the answer is stale."""
    cache = AnswerCache(4)
    base = cache.key("q", "hi", ["c1"])
    cache.put(base, "hindi answer")
    assert cache.get(cache.key("q", "gu", ["c1"])) is None
    assert cache.get(cache.key("q", "hi", ["c2"])) is None
    assert cache.get(cache.key(" q ", "hi", ["c1"])) == "hindi answer"


def test_eviction_is_least_recently_used():
    cache = AnswerCache(2)
    a, b, c = (cache.key(q, "hi", ["s"]) for q in "abc")
    cache.put(a, 1)
    cache.put(b, 2)
    cache.get(a)          # a is now the most recently used
    cache.put(c, 3)       # evicts b, not a
    assert cache.get(a) == 1
    assert cache.get(b) is None
    assert cache.get(c) == 3
    assert cache.stats()["entries"] == 2
