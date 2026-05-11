
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ScoreOutput:
    y_score: np.ndarray
    score_type: str


def extract_scores(estimator, X) -> ScoreOutput:
    """Return continuous score for binary classification if available."""
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return ScoreOutput(y_score=proba[:, 1], score_type="predict_proba")

    if hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(X)
        if np.ndim(decision) == 1:
            return ScoreOutput(y_score=np.asarray(decision), score_type="decision_function")
        if np.ndim(decision) == 2 and decision.shape[1] >= 2:
            return ScoreOutput(y_score=np.asarray(decision)[:, 1], score_type="decision_function")

    return ScoreOutput(y_score=np.full(shape=(len(X),), fill_value=np.nan), score_type="none")


def threshold_predictions(y_score: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous scores to hard predictions using fixed threshold."""
    if np.isnan(y_score).all():
        return np.array([], dtype=int)
    return (y_score >= threshold).astype(int)


def compute_specificity(tn: int, fp: int) -> float:
    """Specificity = TN / (TN + FP), with NaN-safe handling."""
    denom = tn + fp
    if denom == 0:
        return float("nan")
    return float(tn / denom)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute required outer-fold metrics for binary clinical classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: Dict[str, float] = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(compute_specificity(int(tn), int(fp))),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if np.isnan(y_score).all() or len(np.unique(y_true)) < 2:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        return metrics

    metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    return metrics


def bootstrap_median_ci(
    values: np.ndarray,
    confidence_level: float = 0.95,
    n_bootstraps: int = 2000,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Return median and bootstrap CI for the median."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(random_state)
    observed = float(np.median(values))
    boot = np.empty(n_bootstraps, dtype=float)

    for i in range(n_bootstraps):
        sample = rng.choice(values, size=values.size, replace=True)
        boot[i] = np.median(sample)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(boot, alpha / 2.0))
    upper = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return observed, lower, upper


def ci_overlaps(ci_a: Tuple[float, float], ci_b: Tuple[float, float]) -> bool:
    """Whether two confidence intervals overlap."""
    a_lo, a_hi = ci_a
    b_lo, b_hi = ci_b
    if np.isnan([a_lo, a_hi, b_lo, b_hi]).any():
        return False
    return not (a_hi < b_lo or b_hi < a_lo)
