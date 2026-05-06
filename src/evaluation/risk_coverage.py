from __future__ import annotations

import numpy as np


def _validate_binary_or_multiclass_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probs must have shape (n_samples, n_classes).")
    if probs.shape[1] < 2:
        raise ValueError("probs must include at least two classes.")
    return probs


def coverage_from_mask(abstain_mask: np.ndarray) -> float:
    abstain_mask = np.asarray(abstain_mask, dtype=bool)
    return float(1.0 - abstain_mask.mean())


def selective_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    abstain_mask: np.ndarray,
) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    abstain_mask = np.asarray(abstain_mask, dtype=bool)

    keep_mask = ~abstain_mask
    if not np.any(keep_mask):
        return 0.0

    errors = (y_pred[keep_mask] != y_true[keep_mask]).astype(np.float64)
    return float(errors.mean())


def risk_coverage_at_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    probs = _validate_binary_or_multiclass_probs(probs)
    y_true = np.asarray(y_true).ravel().astype(int)

    if len(y_true) != probs.shape[0]:
        raise ValueError("y_true and probs must have the same number of rows.")

    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    abstain_mask = confidences < threshold
    keep_mask = ~abstain_mask

    coverage = float(keep_mask.mean())
    risk = selective_risk(y_true, preds, abstain_mask)
    accuracy = float((preds[keep_mask] == y_true[keep_mask]).mean()) if np.any(keep_mask) else 0.0

    return {
        "threshold": float(threshold),
        "coverage": coverage,
        "risk": risk,
        "accuracy": accuracy,
        "abstention_rate": float(abstain_mask.mean()),
        "retained_count": float(keep_mask.sum()),
    }


def risk_coverage_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> list[dict[str, float]]:
    probs = _validate_binary_or_multiclass_probs(probs)
    y_true = np.asarray(y_true).ravel().astype(int)

    if thresholds is None:
        thresholds = np.unique(np.concatenate(([0.0], probs.max(axis=1), [1.0])))
    thresholds = np.asarray(thresholds, dtype=np.float64)

    return [risk_coverage_at_threshold(probs, y_true, t) for t in thresholds]


def area_under_risk_coverage_curve(curve: list[dict[str, float]]) -> float:
    if not curve:
        raise ValueError("curve must be non-empty.")

    ordered = sorted(curve, key=lambda row: row["coverage"])
    coverage = np.array([row["coverage"] for row in ordered], dtype=np.float64)
    risk = np.array([row["risk"] for row in ordered], dtype=np.float64)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(risk, coverage))
    return float(np.trapz(risk, coverage))
