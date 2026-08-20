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
from core.factory import build_cross_scorer, build_retriever
from core.guardrails.confidence import BASE_FEATURES, CROSS_FEATURE, extract_features
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

    def vector(self, feature_order: tuple[str, ...]) -> list[float]:
        return [self.features[name] for name in feature_order]


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
) -> tuple[list[FeatureSample], tuple[str, ...]]:
    settings = settings or get_settings()
    retriever = build_retriever(settings)
    # The cross-encoder is only a gate feature if it is configured, so the
    # fitted feature set is decided here and recorded with the model.
    cross_scorer = build_cross_scorer(settings)
    feature_order = BASE_FEATURES + ((CROSS_FEATURE,) if cross_scorer else ())
    samples: list[FeatureSample] = []

    for example in load_eval_set(answerable_path):
        sources = retriever.retrieve(example.query, top_k=top_k, language=example.language)
        samples.append(
            FeatureSample(
                query=example.query,
                language=example.language,
                category="answerable",
                label=1,
                features=extract_features(example.query, sources, cross_scorer=cross_scorer),
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
                features=extract_features(row["query"], sources, cross_scorer=cross_scorer),
            )
        )

    return samples, feature_order


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


def _pick_cascade_band(
    cheap_probs: np.ndarray,
    full_probs: np.ndarray,
    cheap_threshold: float,
    full_threshold: float,
) -> tuple[float, float]:
    """Widest band of cheap-model confidence outside which it already agrees.

    Outside the band the cheap model's decision matches the full model's, so
    running the cross-encoder there would change nothing and only cost time.
    """
    cheap_decision = cheap_probs >= cheap_threshold
    full_decision = full_probs >= full_threshold
    disagree = cheap_probs[cheap_decision != full_decision]
    if disagree.size == 0:
        return (cheap_threshold, cheap_threshold)
    # Pad outward so unseen queries near the edges still get verified.
    low = max(0.0, float(disagree.min()) - 0.05)
    high = min(1.0, float(disagree.max()) + 0.05)
    return (low, high)


def repeated_holdout(
    X: np.ndarray,
    y: np.ndarray,
    *,
    splits: int = 30,
    seed: int = 42,
) -> dict[str, float]:
    """Mean held-out metrics over repeated random half-splits."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    runs: list[dict[str, float]] = []
    for offset in range(splits):
        X_fit, X_hold, y_fit, y_hold = train_test_split(
            X, y, test_size=0.5, random_state=seed + offset, stratify=y
        )
        mean = X_fit.mean(axis=0)
        scale = X_fit.std(axis=0)
        scale[scale == 0.0] = 1.0
        model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            (X_fit - mean) / scale, y_fit
        )
        threshold, _ = _pick_operating_point(
            model.predict_proba((X_fit - mean) / scale)[:, 1],
            y_fit,
            np.round(np.arange(0.01, 1.0, 0.005), 4),
        )
        predictions = (model.predict_proba((X_hold - mean) / scale)[:, 1] >= threshold).astype(int)
        runs.append(classification_metrics(predictions, y_hold))

    keys = ("answerable_recall", "false_abstain_rate", "abstain_recall", "balanced_accuracy")
    summary = {k: round(float(np.mean([r[k] for r in runs])), 4) for k in keys}
    summary["balanced_accuracy_stdev"] = round(float(np.std([r["balanced_accuracy"] for r in runs])), 4)
    summary["splits"] = float(splits)
    return summary


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
    samples, feature_order = collect_samples(
        answerable_path=answerable_path,
        abstain_path=abstain_path,
        settings=settings,
        top_k=top_k,
    )
    if not any(s.label == 1 for s in samples) or not any(s.label == 0 for s in samples):
        raise ValueError("need both answerable and abstain samples to calibrate")

    X = np.array([s.vector(feature_order) for s in samples], dtype=float)
    y = np.array([s.label for s in samples], dtype=int)
    X_fit, X_hold, y_fit, y_hold = train_test_split(
        X, y, test_size=0.5, random_state=seed, stratify=y
    )

    # Baseline: threshold the top-1 cosine alone (the first feature).
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

    # Multi-feature logistic gate. Features are standardized first: the cosine
    # and overlap live in [0, 1] while the cross-encoder score spans roughly
    # [-11, +11], and sklearn's L2 penalty is scale-sensitive, so without this
    # the widest feature is the one most shrunk. Mean and scale ship with the
    # model so serving applies the identical transform.
    mean = X_fit.mean(axis=0)
    scale = X_fit.std(axis=0)
    scale[scale == 0.0] = 1.0
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        (X_fit - mean) / scale, y_fit
    )
    p_fit = model.predict_proba((X_fit - mean) / scale)[:, 1]
    p_hold = model.predict_proba((X_hold - mean) / scale)[:, 1]
    t_model, model_fit_metrics = _pick_operating_point(
        p_fit, y_fit, np.round(np.arange(0.01, 1.0, 0.005), 4)
    )
    model_hold_metrics = classification_metrics((p_hold >= t_model).astype(int), y_hold)
    # One half-split leaves ~48 abstain examples to score, so its metrics swing
    # by several points between seeds. The shipped coefficients come from the
    # split above; these repeated splits are what the numbers are quoted from.
    repeated = repeated_holdout(X, y, seed=seed)

    # Cheap pre-gate for the cascade: the same fit without the cross-encoder
    # feature. At serving, a decisive verdict from this model means the
    # cross-encoder is never run. The band is chosen so the cascade's decisions
    # match the full model's; see the sweep in the report.
    cascade_payload = None
    if CROSS_FEATURE in feature_order:
        cheap_cols = [i for i, n in enumerate(feature_order) if n != CROSS_FEATURE]
        Xc_fit, Xc_hold = X_fit[:, cheap_cols], X_hold[:, cheap_cols]
        c_mean, c_scale = Xc_fit.mean(axis=0), Xc_fit.std(axis=0)
        c_scale[c_scale == 0.0] = 1.0
        cheap = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
            (Xc_fit - c_mean) / c_scale, y_fit
        )
        p_cheap_fit = cheap.predict_proba((Xc_fit - c_mean) / c_scale)[:, 1]
        t_cheap, _ = _pick_operating_point(
            p_cheap_fit, y_fit, np.round(np.arange(0.01, 1.0, 0.005), 4)
        )
        band = _pick_cascade_band(p_cheap_fit, p_fit, t_cheap, t_model)
        p_cheap_hold = cheap.predict_proba((Xc_hold - c_mean) / c_scale)[:, 1]
        undecided = (p_cheap_hold > band[0]) & (p_cheap_hold < band[1])
        cascade_payload = {
            "features": [feature_order[i] for i in cheap_cols],
            "coefficients": [float(c) for c in cheap.coef_[0]],
            "intercept": float(cheap.intercept_[0]),
            "feature_mean": [float(v) for v in c_mean],
            "feature_scale": [float(v) for v in c_scale],
            "threshold": float(t_cheap),
            "band": [float(band[0]), float(band[1])],
            "cross_encoder_rate": round(float(undecided.mean()), 4),
        }

    model_payload = {
        "model": "logistic_regression",
        "features": list(feature_order),
        "coefficients": [float(c) for c in model.coef_[0]],
        "intercept": float(model.intercept_[0]),
        "feature_mean": [float(v) for v in mean],
        "feature_scale": [float(v) for v in scale],
        "threshold": float(t_model),
        "target_answerable_recall": TARGET_ANSWERABLE_RECALL,
        "calibration_metrics": model_fit_metrics,
        "held_out_metrics": model_hold_metrics,
    }
    if cascade_payload:
        model_payload["cascade"] = cascade_payload
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
            for i, name in enumerate(feature_order)
        },
        "shipping_default": {
            "feature": feature_order[0],
            "threshold": t_default,
            "held_out": default_hold_metrics,
        },
        "threshold_baseline": {
            "feature": feature_order[0],
            "threshold": t_cosine,
            "calibration": cosine_fit_metrics,
            "held_out": cosine_hold_metrics,
        },
        "cascade": cascade_payload,
        "multi_feature_gate": {
            "coefficients": dict(
                zip(feature_order, [round(float(c), 4) for c in model.coef_[0]], strict=True)
            ),
            "intercept": round(float(model.intercept_[0]), 4),
            "threshold": t_model,
            "calibration": model_fit_metrics,
            "held_out": model_hold_metrics,
            "repeated_holdout": repeated,
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
