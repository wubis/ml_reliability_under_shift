from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss

from src.calibration.temperature import TemperatureScaler
from src.evaluation.risk_coverage import risk_coverage_at_threshold
from src.evaluation.shift_evaluation import summarize_predictions


def run_shift_severity_sweep(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_clean: np.ndarray,
    y_clean: np.ndarray,
    shifted_sets: dict[float, np.ndarray],
    n_bins: int = 10,
    abstain_threshold: float = 0.8,
) -> list[dict[str, float | str]]:
    """
    Evaluate a trained model on a clean set and multiple matched shifted copies.
    """
    clean_probs = model.predict_proba(X_clean)
    clean_logits = model.predict_logits(X_clean)
    cal_logits = model.predict_logits(X_cal)

    temp_scaler = TemperatureScaler().fit(cal_logits, y_cal)
    clean_temp_probs = temp_scaler.predict_proba(clean_logits)

    rows: list[dict[str, float | str]] = []
    clean_summary = summarize_predictions(clean_probs, y_clean, n_bins=n_bins)
    rows.append(
        {
            "split": "clean",
            "severity": 0.0,
            **clean_summary,
            "log_loss": float(log_loss(y_clean, clean_probs)),
            "temp_ece": summarize_predictions(clean_temp_probs, y_clean, n_bins=n_bins)["ece"],
            **{
                f"rc_{k}": v
                for k, v in risk_coverage_at_threshold(clean_probs, y_clean, abstain_threshold).items()
            },
        }
    )

    for severity, X_shift in sorted(shifted_sets.items()):
        shifted_probs = model.predict_proba(X_shift)
        shifted_logits = model.predict_logits(X_shift)
        shifted_temp_probs = temp_scaler.predict_proba(shifted_logits)
        shifted_summary = summarize_predictions(shifted_probs, y_clean, n_bins=n_bins)
        rows.append(
            {
                "split": "shifted",
                "severity": float(severity),
                **shifted_summary,
                "log_loss": float(log_loss(y_clean, shifted_probs)),
                "temp_ece": summarize_predictions(shifted_temp_probs, y_clean, n_bins=n_bins)["ece"],
                **{
                    f"rc_{k}": v
                    for k, v in risk_coverage_at_threshold(shifted_probs, y_clean, abstain_threshold).items()
                },
            }
        )

    baseline = rows[0]
    for row in rows[1:]:
        row["delta_ece"] = float(row["ece"] - baseline["ece"])
        row["delta_brier"] = float(row["brier"] - baseline["brier"])
        row["delta_log_loss"] = float(row["log_loss"] - baseline["log_loss"])
        row["delta_accuracy"] = float(row["accuracy"] - baseline["accuracy"])

    rows[0]["delta_ece"] = 0.0
    rows[0]["delta_brier"] = 0.0
    rows[0]["delta_log_loss"] = 0.0
    rows[0]["delta_accuracy"] = 0.0
    return rows
