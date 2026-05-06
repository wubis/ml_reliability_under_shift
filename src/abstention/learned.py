from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def probability_abstention_features(probs: np.ndarray) -> np.ndarray:
    """
    Build compact uncertainty features from a probability matrix.

    These features are intentionally model-agnostic so the same learned abstainer
    can sit on top of logistic regression or a neural network after calibration.
    """
    probs = _validate_probs(probs)
    sorted_probs = np.sort(probs, axis=1)
    confidence = sorted_probs[:, -1]
    runner_up = sorted_probs[:, -2]
    margin = confidence - runner_up
    entropy = -np.sum(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    normalized_entropy = entropy / np.log(probs.shape[1])
    pred_class = probs.argmax(axis=1).astype(np.float64)

    return np.column_stack([confidence, margin, normalized_entropy, pred_class])


class LearnedAbstainer:
    """
    Learned selective classifier over calibrated probability features.

    The model is trained on held-out calibration predictions to estimate
    ``P(base model is correct | uncertainty features)``. At test time, it
    abstains on examples whose estimated correctness is below a cutoff chosen to
    retain roughly ``target_coverage`` on the calibration split.
    """

    def __init__(self, target_coverage: float = 0.9, random_state: int = 42):
        if not 0.0 < target_coverage <= 1.0:
            raise ValueError("target_coverage must be in (0, 1].")
        self.target_coverage = float(target_coverage)
        self.random_state = random_state
        self.model_: LogisticRegression | None = None
        self.score_cutoff_: float | None = None
        self.constant_score_: float | None = None

    def fit(self, cal_probs: np.ndarray, y_cal: np.ndarray) -> "LearnedAbstainer":
        cal_probs = _validate_probs(cal_probs)
        y_cal = np.asarray(y_cal).ravel().astype(int)
        if len(y_cal) != cal_probs.shape[0]:
            raise ValueError("y_cal and cal_probs must have the same number of rows.")

        correctness = (cal_probs.argmax(axis=1) == y_cal).astype(int)
        features = probability_abstention_features(cal_probs)

        if len(np.unique(correctness)) < 2:
            self.model_ = None
            self.constant_score_ = float(correctness[0])
            scores = np.full(len(y_cal), self.constant_score_, dtype=np.float64)
        else:
            self.model_ = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=self.random_state,
            )
            self.model_.fit(features, correctness)
            scores = self.model_.predict_proba(features)[:, 1]
            self.constant_score_ = None

        abstain_fraction = 1.0 - self.target_coverage
        self.score_cutoff_ = float(np.quantile(scores, abstain_fraction, method="higher"))
        return self

    def correctness_scores(self, probs: np.ndarray) -> np.ndarray:
        if self.score_cutoff_ is None:
            raise ValueError("LearnedAbstainer has not been fit yet.")
        probs = _validate_probs(probs)
        if self.model_ is None:
            return np.full(probs.shape[0], self.constant_score_, dtype=np.float64)
        return self.model_.predict_proba(probability_abstention_features(probs))[:, 1]

    def abstain_mask(self, probs: np.ndarray) -> np.ndarray:
        scores = self.correctness_scores(probs)
        return scores < self.score_cutoff_

    def apply(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        probs = _validate_probs(probs)
        preds = probs.argmax(axis=1)
        scores = self.correctness_scores(probs)
        abstain_mask = scores < self.score_cutoff_
        preds_with_abstain = preds.copy()
        preds_with_abstain[abstain_mask] = -1
        return preds_with_abstain, abstain_mask, scores


def _validate_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probs must have shape (n_samples, n_classes).")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("Each probability row must sum to 1.")
    return probs
