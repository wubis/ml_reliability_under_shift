from __future__ import annotations

import numpy as np


def _as_probability_matrix(prob_predictions: np.ndarray) -> np.ndarray:
    probs = np.asarray(prob_predictions, dtype=np.float64)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.ndim != 2:
        raise ValueError("prob_predictions must be 1D or 2D.")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1.")
    return probs


def brier_score(prob_predictions: np.ndarray, y: np.ndarray) -> float:
    """
    Multi-class Brier score.

    For binary classification, callers may pass either shape (n,) positive-class
    probabilities or shape (n, 2) class probabilities.
    """
    probs = _as_probability_matrix(prob_predictions)
    y = np.asarray(y).ravel().astype(int)

    if len(y) != probs.shape[0]:
        raise ValueError("y and prob_predictions must have the same number of rows.")
    if np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("Labels must be integers in [0, n_classes).")

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def exp_cal_error(
    prob_predictions: np.ndarray,
    y: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error using max-confidence binning.

    This is the standard selective-classification / confidence ECE:
    each example is assigned to a bin based on its predicted confidence
    ``max_k p(y=k | x)``, then we compare average confidence vs accuracy.
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    probs = _as_probability_matrix(prob_predictions)
    y = np.asarray(y).ravel().astype(int)

    if len(y) != probs.shape[0]:
        raise ValueError("y and prob_predictions must have the same number of rows.")
    if np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("Labels must be integers in [0, n_classes).")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, n_bins - 1)

    ece = 0.0
    n = len(y)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)

    return float(ece)
