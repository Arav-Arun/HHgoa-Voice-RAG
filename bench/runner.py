"""Latency benchmarks for the RAG pipeline.

Reports **P50 / P70 / P100** as specified, plus P90/P95/mean/stddev for context,
measured across a large set of real held-out queries rather than one query
repeated.

Three things this deliberately does:

* **Warms up first.** The first call loads a transformer and triggers lazy torch
  init. Including that in the percentiles would report a cold-start artifact as
  though it were steady-state latency; excluding it silently would hide a real
  cost. So it is measured separately and reported as ``cold_start_ms``.
* **Breaks down by stage.** A single total is not verifiable. The per-stage
  table comes from the harness trace, i.e. from the same code that serves
  production requests.
* **Separates the tracks.** The fast path is local and targets 200 ms. The
  quality path makes a remote LLM call and the voice path makes a remote STT
  call; neither can be 200 ms, and averaging them together would misrepresent
  both.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

from core.config import Settings, get_settings

DEFAULT_OUTPUT_PATH = Path("data/bench/latency.json")
DEFAULT_QUERIES_PATH = Path("data/eval/queries.jsonl")
DEFAULT_ABSTAIN_PATH = Path("data/eval/abstain_queries.jsonl")


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. P100 is the maximum, by definition."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if pct >= 100:
        return ordered[-1]
    idx = round((pct / 100.0) * (len(ordered) - 1))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "p50_ms": round(percentile(values, 50), 2),
        "p70_ms": round(percentile(values, 70), 2),
        "p90_ms": round(percentile(values, 90), 2),
        "p95_ms": round(percentile(values, 95), 2),
        "p100_ms": round(percentile(values, 100), 2),
        "mean_ms": round(statistics.mean(values), 2),
        "stddev_ms": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0,
        "min_ms": round(min(values), 2),
    }


def load_bench_queries(
    *,
    queries_path: Path = DEFAULT_QUERIES_PATH,
    abstain_path: Path = DEFAULT_ABSTAIN_PATH,
    limit: int | None = None,
    include_abstain: bool = True,
) -> list[tuple[str, str, str]]:
    """Return (query, language, kind) triples.

    Abstain and adversarial queries are included on purpose: they exercise the
    guardrail short-circuits, and a benchmark that only measures the happy path
    understates the branch mix a real deployment sees.
    """
    rows: list[tuple[str, str, str]] = []

    if queries_path.exists():
        with queries_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append((row["query"], row.get("language", "hi"), "answerable"))

    if limit is not None:
        rows = rows[:limit]

    if include_abstain and abstain_path.exists():
        with abstain_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append((row["query"], row.get("language", "hi"), row.get("category", "abstain")))

    return rows


def _host_info(settings: Settings, chunk_count: int) -> dict:
    import numpy as np

    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "embedding_preset": settings.embedding_preset,
        "chunking_provider": settings.chunking_provider,
        "retriever": settings.retriever_provider,
        "fusion": settings.fusion_method,
        "top_k": settings.top_k,
        "indexed_chunks": chunk_count,
    }
    try:
        import torch

        info["torch_threads"] = torch.get_num_threads()
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - torch always present
        # Thread count is reporting metadata, not required for the benchmark;
        # record why it is missing rather than dropping it silently.
        info["torch_threads"] = f"unavailable: {type(exc).__name__}"
    return info


def bench_pipeline(
    queries: list[tuple[str, str, str]],
    *,
    mode: str = "fast",
    warmup: int = 3,
    top_k: int | None = None,
    settings: Settings | None = None,
) -> dict:
    """Time the pipeline over a query set, with per-stage breakdown."""
    from core.factory import build_rag_pipeline

    settings = settings or get_settings()
    build_start = time.perf_counter()
    pipeline = build_rag_pipeline(settings, local_only=(mode == "fast"))
    build_ms = (time.perf_counter() - build_start) * 1000.0

    if not queries:
        return {"error": "no queries"}

    # Cold start = time until the service can answer at steady-state speed, so
    # it must force every lazily-loaded model, not just the ones the first
    # query happens to touch. Measured once and excluded from the percentiles.
    cold_start = time.perf_counter()
    pipeline.warm()
    cold_ms = (time.perf_counter() - cold_start) * 1000.0

    for query, language, _ in queries[:warmup]:
        pipeline.query(query, language=language, mode=mode)

    totals: list[float] = []
    by_stage: dict[str, list[float]] = {}
    by_kind: dict[str, list[float]] = {}
    by_language: dict[str, list[float]] = {}
    paths: dict[str, int] = {}

    for query, language, kind in queries:
        start = time.perf_counter()
        response = pipeline.query(query, language=language, top_k=top_k, mode=mode)
        elapsed = (time.perf_counter() - start) * 1000.0

        totals.append(elapsed)
        by_kind.setdefault(kind, []).append(elapsed)
        by_language.setdefault(language, []).append(elapsed)
        path = response.metadata.get("path", "unknown")
        paths[path] = paths.get(path, 0) + 1
        for stage, ms in response.metadata.get("timings_ms", {}).items():
            by_stage.setdefault(stage, []).append(float(ms))

    return {
        "mode": mode,
        "queries": len(queries),
        "pipeline_build_ms": round(build_ms, 2),
        "cold_start_ms": round(cold_ms, 2),
        "total": summarize(totals),
        "by_stage": {name: summarize(vals) for name, vals in sorted(by_stage.items())},
        "by_kind": {name: summarize(vals) for name, vals in sorted(by_kind.items())},
        "by_language": {name: summarize(vals) for name, vals in sorted(by_language.items())},
        "paths": paths,
    }


def bench_voice(
    audio_paths: list[Path],
    *,
    language: str = "hi",
    mode: str = "fast",
    settings: Settings | None = None,
) -> dict:
    """End-to-end voice latency, with the STT round trip broken out.

    Run against a small N: every call is a paid network request, and the point
    is to show STT's share of the wall clock, not to gather a tight percentile.
    """
    from core.factory import build_rag_pipeline, build_stt

    settings = settings or get_settings()
    stt = build_stt()
    pipeline = build_rag_pipeline(settings, local_only=(mode == "fast"))

    stt_ms: list[float] = []
    rag_ms: list[float] = []
    total_ms: list[float] = []
    transcripts: list[str] = []

    for path in audio_paths:
        audio = path.read_bytes()
        start = time.perf_counter()
        result = stt.transcribe(audio, language=language, content_type="audio/wav", filename=path.name)
        after_stt = time.perf_counter()
        pipeline.query(result.text, language=language, mode=mode)
        end = time.perf_counter()

        stt_ms.append((after_stt - start) * 1000.0)
        rag_ms.append((end - after_stt) * 1000.0)
        total_ms.append((end - start) * 1000.0)
        transcripts.append(result.text)

    return {
        "clips": len(audio_paths),
        "language": language,
        "stt_provider": settings.stt_provider,
        "stt_model": settings.stt_model,
        "stt": summarize(stt_ms),
        "rag": summarize(rag_ms),
        "end_to_end": summarize(total_ms),
        "transcripts": transcripts,
    }


def run_benchmarks(
    *,
    limit: int | None = None,
    modes: tuple[str, ...] = ("fast",),
    audio_dir: Path | None = None,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
    settings: Settings | None = None,
) -> dict:
    settings = settings or get_settings()
    queries = load_bench_queries(limit=limit)

    from core.factory import build_vector_store

    chunk_count = build_vector_store(settings).count()
    report: dict = {
        "host": _host_info(settings, chunk_count),
        "target_ms": 200,
        "tracks": {},
    }

    for mode in modes:
        report["tracks"][mode] = bench_pipeline(queries, mode=mode, settings=settings)

    if audio_dir is not None and audio_dir.exists():
        clips = sorted(audio_dir.glob("*.wav"))
        if clips:
            report["tracks"]["voice"] = bench_voice(clips, settings=settings)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report
