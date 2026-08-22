"""Serving counters: what this process has actually answered.

The page already reports two things and they answer different questions. The
metric cards come from committed benchmark runs, reproducible with
``uv run hhgoa bench``. The session panel counts one browser tab, which is too few
queries to mean anything. Neither says what the deployed service has done.

This is the third: every query the process has served, held in memory. Not
persisted, because the container's disk is ephemeral and a number that silently
resets on redeploy is worse than one that plainly counts from startup. The
label says "since the service started" for that reason.

Bounded on purpose. A demo left open should not grow a list without limit, and
percentiles over the most recent window describe current behaviour better than
percentiles over every query since boot.
"""

from __future__ import annotations

import threading
from collections import deque

# Enough that percentiles are meaningful, small enough to stay trivial in
# memory next to a 1.16 GiB index.
WINDOW = 2000


class ServingCounters:
    def __init__(self, window: int = WINDOW) -> None:
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self._stages: dict[str, deque[float]] = {}
        self._window = window
        self.total = 0
        self.abstained = 0
        self.voice = 0

    def record(
        self,
        total_ms: float,
        *,
        abstained: bool,
        stages: dict[str, float] | None = None,
        voice: bool = False,
    ) -> None:
        with self._lock:
            self.total += 1
            self.abstained += bool(abstained)
            self.voice += bool(voice)
            self._latencies.append(float(total_ms))
            for name, ms in (stages or {}).items():
                bucket = self._stages.get(name)
                if bucket is None:
                    bucket = self._stages[name] = deque(maxlen=self._window)
                bucket.append(float(ms))

    @staticmethod
    def _percentile(ordered: list[float], q: float) -> float | None:
        if not ordered:
            return None
        # Nearest-rank, the same rule bench/runner.py uses, so a reader can put
        # these numbers beside the published ones without adjusting for method.
        return ordered[min(int(len(ordered) * q), len(ordered) - 1)]

    def snapshot(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies)
            stages = {name: sorted(values) for name, values in self._stages.items()}
            total, abstained, voice = self.total, self.abstained, self.voice

        return {
            "queries": total,
            "abstained": abstained,
            "voice_queries": voice,
            "sampled": len(latencies),
            "p50_ms": self._percentile(latencies, 0.5),
            "p70_ms": self._percentile(latencies, 0.7),
            "p100_ms": latencies[-1] if latencies else None,
            "histogram": self._histogram(latencies),
            # Median rather than mean: one cross-encoder escalation drags a mean
            # and misreports what a typical query spends in that stage.
            "stages": {
                name: {"p50_ms": self._percentile(values, 0.5), "count": len(values)}
                for name, values in stages.items()
            },
        }

    @staticmethod
    def _histogram(ordered: list[float]) -> list[dict]:
        # Fixed edges. Computed ones would reshape the distribution as queries
        # arrive, which makes the chart unreadable over a demo.
        edges = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 400]
        counts = [0] * (len(edges) + 1)
        for ms in ordered:
            index = next((i for i, edge in enumerate(edges) if ms <= edge), len(edges))
            counts[index] += 1
        return [
            {"le": edges[i] if i < len(edges) else None, "count": count}
            for i, count in enumerate(counts)
        ]
