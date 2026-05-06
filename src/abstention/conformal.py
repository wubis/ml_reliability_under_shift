from __future__ import annotations

import numpy as np


class SplitConformalAbstainer:
    """
    Split-conformal prediction sets using score 1 - P(y_true | x).

    A prediction is retained when the conformal set has exactly one label. Empty
    or multi-label sets are abstained because the model is not making a unique,
    coverage-controlled decision.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1).")
        self.alpha = float(alpha)
        self.qhat_: float | None = None

    def fit(self, cal_probs: np.ndarray, y_cal: np.ndarray) -> "SplitConformalAbstainer":
        cal_probs = self._validate_probs(cal_probs)
        y_cal = np.asarray(y_cal).ravel().astype(int)
        if len(y_cal) != cal_probs.shape[0]:
            raise ValueError("y_cal and cal_probs must have the same number of rows.")

        scores = 1.0 - cal_probs[np.arange(len(y_cal)), y_cal]
        n = len(scores)
        quantile_level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        quantile_level = min(1.0, quantile_level)
        self.qhat_ = float(np.quantile(scores, quantile_level, method="higher"))
        return self

    def prediction_sets(self, probs: np.ndarray) -> np.ndarray:
        if self.qhat_ is None:
            raise ValueError("SplitConformalAbstainer has not been fit yet.")
        probs = self._validate_probs(probs)
        return (1.0 - probs) <= self.qhat_

    def abstain_mask(self, probs: np.ndarray) -> np.ndarray:
        sets = self.prediction_sets(probs)
        return sets.sum(axis=1) != 1

    def apply(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = self._validate_probs(probs)
        sets = self.prediction_sets(probs)
        abstain_mask = sets.sum(axis=1) != 1
        preds = probs.argmax(axis=1)
        preds_with_abstain = preds.copy()
        preds_with_abstain[abstain_mask] = -1
        return preds_with_abstain, abstain_mask, sets

    @staticmethod
    def _validate_probs(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError("probs must have shape (n_samples, n_classes).")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each probability row must sum to 1.")
        return probs
