"""Calibrate the grounding gate from answerable vs unanswerable queries.

Fits the multi-feature gate used at serving time, and reports it against the
single-threshold baseline it replaces.

Three methodological points worth stating, because each one changes the numbers:

1. **Only gate-reaching traffic is used.** Unsafe and prompt-injection queries
   are refused by the input-intent filter before retrieval happens, so
   including them here would credit the retrieval-confidence gate with
   refusals it never made.
2. **Fit and report are split.** Coefficients and the operating point come from
   one half; every reported metric comes from the other. The previous version
   of this file chose and reported a threshold on the same 24 rows.
3. **"Balanced accuracy" means the mean of the two class recalls.** The
   previous version reported ``min(recall_answerable, recall_abstain)`` under
   that name, which is worst-class recall and is always the lower number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.config import Settings, get_settings
from core.factory import build_retriever
from core.guardrails.confidence import FEATURE_ORDER, extract_features
from eval.dataset import load_eval_set

DEFAULT_ANSWERABLE_PATH = Path("data/eval/queries.jsonl")
DEFAULT_ABSTAIN_PATH = Path("data/eval/abstain_queries.jsonl")
DEFAULT_OUTPUT_PATH = Path("data/eval/guardrail-calibration.json")
DEFAULT_MODEL_PATH = Path("data/eval/guardrail-model.json")

# Abstain categories that actually reach the grounding gate.
GATE_CATEGORIES = ("off_topic", "nonsense", "in_domain_unanswerable")
TARGET_ANSWERABLE_RECALL = 0.95


@dataclass
class FeatureSample:
    query: str
    language: str
    category: str
    label: int  # 1 = answerable, 0 = should abstain
    features: dict[str, float]

    def vector(self) -> list[float]:
        return [self.features[name] for name in FEATURE_ORDER]


def _load_abstain_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def collect_samples(
    *,
    answerable_path: Path = DEFAULT_ANSWERABLE_PATH,
    abstain_path: Path = DEFAULT_ABSTAIN_PATH,
    settings: Settings | None = None,
    top_k: int = 5,
) -> list[FeatureSample]:
    settings = settings or get_settings()
    retriever = build_retriever(settings)
    samples: list[FeatureSample] = []

    for example in load_eval_set(answerable_path):
        sources = retriever.retrieve(example.query, top_k=top_k, language=example.language)
        samples.append(
            FeatureSample(
                query=example.query,
                language=example.language,
                category="answerable",
                label=1,
                features=extract_features(example.query, sources),
            )
        )

    for row in _load_abstain_rows(abstain_path):
        if row.get("category") not in GATE_CATEGORIES:
            continue
        language = row.get("language", "hi")
        sources = retriever.retrieve(row["query"], top_k=top_k, language=language)
        samples.append(
            FeatureSample(
                query=row["query"],
                language=language,
                category=row["category"],
                label=0,
                features=extract_features(row["query"], sources),
            )
        )

    return samples


def classification_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    tp = int(((pred == 1) & (truth == 1)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    tn = int(((pred == 0) & (truth == 0)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())

    answerable_recall = tp / (tp + fn) if tp + fn else 0.0
    abstain_recall = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * answerable_recall / (precision + answerable_recall) if precision + answerable_recall else 0.0

    return {
        "answerable_recall": round(answerable_recall, 4),
        "abstain_recall": round(abstain_recall, 4),
        "answerable_precision": round(precision, 4),
        "answerable_f1": round(f1, 4),
        # Mean of the two class recalls, the actual definition.
        "balanced_accuracy": round((answerable_recall + abstain_recall) / 2, 4),
        # What the previous implementation mislabelled as balanced accuracy.
        "worst_class_recall": round(min(answerable_recall, abstain_recall), 4),
        "false_abstain_rate": round(fn / (tp + fn), 4) if tp + fn else 0.0,
        "true_positives": tp,
        "false_negatives": fn,
        "true_negatives": tn,
        "false_positives": fp,
    }


def _pick_operating_point(
    scores_fit: np.ndarray,
    labels_fit: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, dict]:
    """Maximize abstain recall subject to answerable recall >= target.

    Refusing a real question is the more damaging error for this product, so the
    answerable-recall floor is a constraint rather than something traded off.
    """
    best: tuple[float, dict] | None = None
    for threshold in grid:
        metrics = classification_metrics((scores_fit >= threshold).astype(int), labels_fit)
        if metrics["answerable_recall"] >= TARGET_ANSWERABLE_RECALL and (
            best is None or metrics["abstain_recall"] > best[1]["abstain_recall"]
        ):
            best = (float(threshold), metrics)
    if best is None:
        best = max(
            (
                (float(t), classification_metrics((scores_fit >= t).astype(int), labels_fit))
                for t in grid
            ),
            key=lambda kv: kv[1]["balanced_accuracy"],
        )
    return best


def calibrate_and_write(
    *,
    answerable_path: Path = DEFAULT_ANSWERABLE_PATH,
    abstain_path: Path = DEFAULT_ABSTAIN_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    settings: Settings | None = None,
    top_k: int = 5,
    seed: int = 42,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    settings = settings or get_settings()
    samples = collect_samples(
        answerable_path=answerable_path,
        abstain_path=abstain_path,
        settings=settings,
        top_k=top_k,
    )
    if not any(s.label == 1 for s in samples) or not any(s.label == 0 for s in samples):
        raise ValueError("need both answerable and abstain samples to calibrate")

    X = np.array([s.vector() for s in samples], dtype=float)
    y = np.array([s.label for s in samples], dtype=int)
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    # Baseline: threshold the top-1 cosine alone (index 0 of FEATURE_ORDER).
    cosine_grid = np.round(np.arange(0.60, 0.99, 0.001), 4)
    t_cosine, cosine_fit_metrics = _pick_operating_point(X_fit[:, 0], y_fit, cosine_grid)
    cosine_hold_metrics = classification_metrics((X_hold[:, 0] >= t_cosine).astype(int), y_hold)
    # What the configured default actually does, reported on the same held-out
    # half. This is the gate a fresh checkout runs with when no fitted model is
    # present, so its behaviour belongs in the report rather than in folklore.
    t_default = settings.guardrail_min_score
    default_hold_metrics = classification_metrics(
        (X_hold[:, 0] >= t_default).astype(int), y_hold
    )

    # Multi-feature logistic gate.
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_fit, y_fit)
    p_fit = model.predict_proba(X_fit)[:, 1]
    p_hold = model.predict_proba(X_hold)[:, 1]
    t_model, model_fit_metrics = _pick_operating_point(
        p_fit, y_fit, np.round(np.arange(0.01, 1.0, 0.005), 4)
    )
    model_hold_metrics = classification_metrics((p_hold >= t_model).astype(int), y_hold)

    model_payload = {
        "model": "logistic_regression",
        "features": list(FEATURE_ORDER),
        "coefficients": [float(c) for c in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "threshold": float(t_model),
        "target_answerable_recall": TARGET_ANSWERABLE_RECALL,
        "calibration_metrics": model_fit_metrics,
        "held_out_metrics": model_hold_metrics,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")

    by_category: dict[str, int] = {}
    for sample in samples:
        by_category[sample.category] = by_category.get(sample.category, 0) + 1

    report = {
        "embedding_preset": settings.embedding_preset,
        "retriever": settings.retriever_provider,
        "top_k": top_k,
        "samples": {"total": len(samples), "by_category": by_category},
        "gate_categories": list(GATE_CATEGORIES),
        "target_answerable_recall": TARGET_ANSWERABLE_RECALL,
        "split": {"fit": len(y_fit), "held_out": len(y_hold), "seed": seed},
        "feature_separation": {
            name: {
                "answerable_mean": round(float(X[y == 1][:, i].mean()), 4),
                "abstain_mean": round(float(X[y == 0][:, i].mean()), 4),
            }
            for i, name in enumerate(FEATURE_ORDER)
        },
        "shipping_default": {
            "feature": FEATURE_ORDER[0],
            "threshold": t_default,
            "held_out": default_hold_metrics,
        },
        "threshold_baseline": {
            "feature": FEATURE_ORDER[0],
            "threshold": t_cosine,
            "calibration": cosine_fit_metrics,
            "held_out": cosine_hold_metrics,
        },
        "multi_feature_gate": {
            "coefficients": dict(
                zip(FEATURE_ORDER, [round(float(c), 4) for c in model.coef_[0]], strict=True)
            ),
            "intercept": round(float(model.intercept_[0]), 4),
            "threshold": t_model,
            "calibration": model_fit_metrics,
            "held_out": model_hold_metrics,
        },
        "model_path": str(model_path),
        "note": (
            "Calibrated only on abstain categories reaching the grounding gate; unsafe "
            "and prompt-injection queries are refused earlier by the input-intent filter. "
            "balanced_accuracy is the mean of the two class recalls; worst_class_recall is "
            "the minimum (what earlier revisions reported under the 'balanced accuracy' name)."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(calibrate_and_write(), indent=2, ensure_ascii=False))
