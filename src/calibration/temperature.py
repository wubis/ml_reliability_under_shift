from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from src.calibration.base import BaseCalibrator


def _ensure_logit_matrix(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
    if logits.ndim != 2:
        raise ValueError("logits must be 1D or 2D.")
    return logits


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


class TemperatureScaler(BaseCalibrator):
    """
    Post-hoc temperature scaling for binary or multiclass logits.

    Binary logits may be shape ``(n,)`` or ``(n, 1)``. Multiclass logits should
    be shape ``(n, K)``.
    """

    def __init__(self, temperature: float = 1.0, bounds: tuple[float, float] = (1e-3, 100.0)):
        self.temperature = float(temperature)
        self.bounds = bounds
        self.is_fit = False

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        logits = _ensure_logit_matrix(logits)
        y = np.asarray(y).ravel().astype(int)

        if len(y) != logits.shape[0]:
            raise ValueError("y and logits must have the same number of rows.")

        def objective(temp: float) -> float:
            scaled_logits = logits / temp
            if scaled_logits.shape[1] == 1:
                probs_pos = _sigmoid(scaled_logits[:, 0])
                probs_pos = np.clip(probs_pos, 1e-12, 1.0 - 1e-12)
                return float(-np.mean(y * np.log(probs_pos) + (1 - y) * np.log(1.0 - probs_pos)))

            probs = _softmax(scaled_logits)
            probs = np.clip(probs, 1e-12, 1.0)
            return float(-np.mean(np.log(probs[np.arange(len(y)), y])))

        result = minimize_scalar(objective, bounds=self.bounds, method="bounded")
        self.temperature = float(result.x)
        self.is_fit = True
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise ValueError("TemperatureScaler has not been fit yet.")
        logits = _ensure_logit_matrix(logits)
        return logits / self.temperature

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        scaled_logits = self.transform_logits(logits)
        if scaled_logits.shape[1] == 1:
            probs_pos = _sigmoid(scaled_logits[:, 0])
            probs_neg = 1.0 - probs_pos
            return np.column_stack([probs_neg, probs_pos])
        return _softmax(scaled_logits)
