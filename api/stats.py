"""Read measured results off disk for the demo UI.

Every number the UI shows comes from an artifact written by an actual run
(``./hhgoa retriever-compare``, ``./hhgoa bench``, ``./hhgoa guardrail-calibrate``).
Nothing is hardcoded. A missing artifact yields ``None`` and the UI renders a
dash, so an unmeasured metric is visibly absent rather than quietly invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RETRIEVER_PATH = Path("data/eval/retriever-compare.json")
BENCH_PATH = Path("data/bench/latency.json")
# The calibration *report*, not the fitted model. The model file records one
# half-split, which swings by several points between seeds; the report carries
# the 30-split average that the README quotes. The page must not disagree with
# the docs.
GUARDRAIL_PATH = Path("data/eval/guardrail-calibration.json")


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _dig(data: dict | None, *keys: str) -> Any:
    """Walk nested keys, returning None if any level is missing."""
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return None
        data = data[key]
    return data


def collect_stats() -> dict[str, Any]:
    retriever = _load(RETRIEVER_PATH)
    bench = _load(BENCH_PATH)
    guardrail = _load(GUARDRAIL_PATH)

    hybrid = _dig(retriever, "hybrid", "overall")
    dense = _dig(retriever, "dense", "overall")
    latency = _dig(bench, "tracks", "fast", "total")
    voice = _dig(bench, "tracks", "voice")
    gate = _dig(guardrail, "multi_feature_gate", "repeated_holdout")

    return {
        "retrieval": {
            "hit_at_5": _dig(hybrid, "hit@5"),
            "mrr": _dig(hybrid, "mrr"),
            "baseline_hit_at_5": _dig(dense, "hit@5"),
            "queries": _dig(hybrid, "count"),
        },
        "latency": {
            "p50_ms": _dig(latency, "p50_ms"),
            "p70_ms": _dig(latency, "p70_ms"),
            "p100_ms": _dig(latency, "p100_ms"),
            "queries": _dig(latency, "count"),
            "budget_ms": _dig(bench, "target_ms"),
            # The machine these percentiles were measured on. A deployment runs
            # on slower shared CPU, so the page must not imply one number
            # describes both.
            "platform": _dig(bench, "host", "platform"),
        },
        # Speech-to-text is reported apart from the pipeline percentiles: it is a
        # network call to ElevenLabs and sits outside the 200 ms budget.
        "voice": {
            "stt_p50_ms": _dig(voice, "stt", "p50_ms"),
            "rag_p50_ms": _dig(voice, "rag", "p50_ms"),
            "end_to_end_p50_ms": _dig(voice, "end_to_end", "p50_ms"),
            "clips": _dig(voice, "stt", "count"),
        },
        "guardrail": {
            # Reported as a pair on purpose: a false-abstain rate alone says
            # nothing about what the refusals bought.
            "false_abstain_rate": _dig(gate, "false_abstain_rate"),
            "abstain_recall": _dig(gate, "abstain_recall"),
        },
        "sources": [
            str(path)
            for path, data in (
                (RETRIEVER_PATH, retriever),
                (BENCH_PATH, bench),
                (GUARDRAIL_PATH, guardrail),
            )
            if data
        ],
    }
