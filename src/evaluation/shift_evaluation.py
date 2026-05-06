from __future__ import annotations

import numpy as np

from src.evaluation.calibration_metrics import brier_score, exp_cal_error


def summarize_predictions(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y).ravel().astype(int)
    preds = probs.argmax(axis=1)

    return {
        "accuracy": float((preds == y).mean()),
        "ece": exp_cal_error(probs, y, n_bins=n_bins),
        "brier": brier_score(probs, y),
        "avg_confidence": float(probs.max(axis=1).mean()),
    }


def evaluate_shift_degradation(
    clean_probs: np.ndarray,
    shifted_probs: np.ndarray,
    y_clean: np.ndarray,
    y_shifted: np.ndarray | None = None,
    n_bins: int = 10,
) -> dict[str, dict[str, float]]:
    """
    Compare the same model's predictive reliability on clean vs shifted data.
    """
    if y_shifted is None:
        y_shifted = y_clean

    clean = summarize_predictions(clean_probs, y_clean, n_bins=n_bins)
    shifted = summarize_predictions(shifted_probs, y_shifted, n_bins=n_bins)

    delta = {
        key: float(shifted[key] - clean[key])
        for key in clean
    }

    return {
        "clean": clean,
        "shifted": shifted,
        "delta_shift_minus_clean": delta,
    }
