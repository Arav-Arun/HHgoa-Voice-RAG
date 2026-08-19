"""Calibrate GUARDRAIL_MIN_SCORE from answerable vs abstain query score distributions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.config import Settings, get_settings
from core.factory import build_retriever
from eval.dataset import load_eval_set

DEFAULT_ANSWERABLE_PATH = Path("data/eval/queries.jsonl")
DEFAULT_ABSTAIN_PATH = Path("data/eval/abstain_queries.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/guardrail-calibration.json")


@dataclass
class ScoreSample:
    query: str
    language: str
    best_score: float
    label: str  # answerable | abstain


def _best_scores(
    queries: list[tuple[str, str]],
    *,
    settings: Settings | None = None,
    top_k: int = 5,
) -> list[float]:
    retriever = build_retriever(settings)
    scores: list[float] = []
    for query, _lang in queries:
        results = retriever.retrieve(query, top_k=top_k)
        scores.append(max((row.score for row in results), default=0.0))
    return scores


def collect_score_samples(
    *,
    answerable_path: Path = DEFAULT_ANSWERABLE_PATH,
    abstain_path: Path = DEFAULT_ABSTAIN_PATH,
    settings: Settings | None = None,
    top_k: int = 5,
) -> list[ScoreSample]:
    settings = settings or get_settings()
    answerable = load_eval_set(answerable_path)
    abstain_rows = load_eval_set(abstain_path)

    samples: list[ScoreSample] = []
    answerable_queries = [(ex.query, ex.language) for ex in answerable]
    abstain_queries = [(ex.query, ex.language) for ex in abstain_rows]

    for (query, lang), score in zip(
        answerable_queries,
        _best_scores(answerable_queries, settings=settings, top_k=top_k),
        strict=True,
    ):
        samples.append(ScoreSample(query=query, language=lang, best_score=score, label="answerable"))

    for (query, lang), score in zip(
        abstain_queries,
        _best_scores(abstain_queries, settings=settings, top_k=top_k),
        strict=True,
    ):
        samples.append(ScoreSample(query=query, language=lang, best_score=score, label="abstain"))

    return samples


def recommend_threshold(samples: list[ScoreSample]) -> dict:
    answerable = [s.best_score for s in samples if s.label == "answerable"]
    abstain = [s.best_score for s in samples if s.label == "abstain"]

    if not answerable or not abstain:
        raise ValueError("Need both answerable and abstain samples to calibrate")

    thresholds = [round(i * 0.01, 2) for i in range(0, 101)]
    best_threshold = 0.0
    best_score = -1.0
    rows: list[dict] = []

    for threshold in thresholds:
        answerable_ok = sum(score >= threshold for score in answerable) / len(answerable)
        abstain_ok = sum(score < threshold for score in abstain) / len(abstain)
        balanced = min(answerable_ok, abstain_ok)
        rows.append(
            {
                "threshold": threshold,
                "answerable_recall": round(answerable_ok, 4),
                "abstain_recall": round(abstain_ok, 4),
                "balanced_accuracy": round(balanced, 4),
            }
        )
        if balanced > best_score:
            best_score = balanced
            best_threshold = threshold

    return {
        "recommended_min_score": best_threshold,
        "balanced_accuracy": round(best_score, 4),
        "answerable_count": len(answerable),
        "abstain_count": len(abstain),
        "answerable_scores": {
            "min": round(min(answerable), 4),
            "p25": round(_percentile(answerable, 0.25), 4),
            "median": round(_percentile(answerable, 0.5), 4),
            "p75": round(_percentile(answerable, 0.75), 4),
            "max": round(max(answerable), 4),
        },
        "abstain_scores": {
            "min": round(min(abstain), 4),
            "p25": round(_percentile(abstain, 0.25), 4),
            "median": round(_percentile(abstain, 0.5), 4),
            "p75": round(_percentile(abstain, 0.75), 4),
            "max": round(max(abstain), 4),
        },
        "threshold_sweep": rows,
    }


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = p * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def calibrate_and_write(
    *,
    answerable_path: Path = DEFAULT_ANSWERABLE_PATH,
    abstain_path: Path = DEFAULT_ABSTAIN_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    settings: Settings | None = None,
    top_k: int = 5,
) -> dict:
    settings = settings or get_settings()
    samples = collect_score_samples(
        answerable_path=answerable_path,
        abstain_path=abstain_path,
        settings=settings,
        top_k=top_k,
    )
    report = recommend_threshold(samples)
    report["embedding_provider"] = settings.embedding_provider
    report["embedding_preset"] = settings.embedding_preset
    report["top_k"] = top_k
    report["answerable_path"] = str(answerable_path)
    report["abstain_path"] = str(abstain_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = calibrate_and_write()
    print(json.dumps(result, indent=2, ensure_ascii=False))
