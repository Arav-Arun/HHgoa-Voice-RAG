"""Latency / throughput benchmarks."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

from core.factory import build_rag_pipeline


@dataclass
class BenchResult:
    operation: str
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100) * (len(ordered) - 1)))
    return ordered[idx]


def bench_query(query: str, *, runs: int = 10) -> BenchResult:
    pipeline = build_rag_pipeline(use_template_llm=True)
    timings: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()
        pipeline.query(query)
        timings.append((time.perf_counter() - start) * 1000)

    return BenchResult(
        operation="query",
        runs=runs,
        mean_ms=statistics.mean(timings),
        p50_ms=_percentile(timings, 50),
        p95_ms=_percentile(timings, 95),
    )
