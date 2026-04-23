"""
Baseline logistic regression classifier.

Wraps ``sklearn.linear_model.LogisticRegression`` in a sklearn-style class whose
public API matches ``NeuralNetClassifier`` in ``baseline_nn.py``, so downstream
calibration and abstention code can use either baseline interchangeably.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


# Default inverse-regularization grid for cross-validation (C = 1 / lambda).
_DEFAULT_CS: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def _select_solver(penalty: str) -> str:
    """L-BFGS for L2 / unregularized, liblinear for L1."""
    if penalty == "l1":
        return "liblinear"
    return "lbfgs"


class LogisticRegressionClassifier:
    """
    sklearn-like wrapper around ``LogisticRegression``.

    Public API (matches ``NeuralNetClassifier``):
        - ``fit(X, y)`` -> self
        - ``predict_logits(X)`` -> shape ``(n,)``
        - ``predict_proba(X)`` -> shape ``(n, 2)`` with columns ``[P(y=0), P(y=1)]``
        - ``predict(X, threshold=0.5)`` -> shape ``(n,)``
        - ``get_params()`` -> dict
        - ``clone()`` -> deep copy

    Extra helpers:
        - ``get_coefficients()`` -> ``(w, b)`` for interpretability plots
        - ``nonconformity_scores(X, y)`` -> ``1 - P(y_true | x)`` for split conformal
    """

    def __init__(
        self,
        penalty: str = "l2",
        C: Optional[float] = None,
        Cs: Optional[Sequence[float]] = None,
        cv: int = 5,
        max_iter: int = 1000,
        random_state: int = 42,
        scoring: str = "neg_log_loss",
    ):
        if penalty not in {"l1", "l2", "none"}:
            raise ValueError(
                f"penalty must be one of 'l1', 'l2', 'none'; got {penalty!r}"
            )

        self.penalty = penalty
        self.C = C
        self.Cs = tuple(Cs) if Cs is not None else _DEFAULT_CS
        self.cv = cv
        self.max_iter = max_iter
        self.random_state = random_state
        self.scoring = scoring

        # Populated by fit().
        self.model: Optional[LogisticRegression] = None
        self.classes_: Optional[np.ndarray] = None
        self.selected_C_: Optional[float] = None
        # Kept for API parity with the NN baseline. LR uses a deterministic solver
        # on a convex objective, so there is no per-epoch loss curve to record.
        self.loss_history: list[float] = []

    # ---------------------------------------------------------------- fit

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        """Fit on (X, y). If ``self.C is None``, pick C by K-fold CV on log-loss."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        solver = _select_solver(self.penalty)
        sk_penalty = None if self.penalty == "none" else self.penalty

        if self.C is None:
            # LogisticRegressionCV does not accept penalty=None; if the caller asked
            # for no regularization we fall through to a weak-L2 proxy via a large C.
            cv_penalty = sk_penalty if sk_penalty is not None else "l2"
            cv_model = LogisticRegressionCV(
                Cs=list(self.Cs),
                cv=self.cv,
                penalty=cv_penalty,
                solver=solver,
                scoring=self.scoring,
                max_iter=self.max_iter,
                random_state=self.random_state,
                refit=True,
                n_jobs=None,
            )
            cv_model.fit(X, y)
            self.selected_C_ = float(np.atleast_1d(cv_model.C_)[0])
            self.model = cv_model
        else:
            self.selected_C_ = float(self.C)
            self.model = LogisticRegression(
                penalty=sk_penalty,
                C=self.C,
                solver=solver,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            self.model.fit(X, y)

        self.classes_ = self.model.classes_
        return self

    # ------------------------------------------------------------ predict

    def _check_fitted(self) -> None:
        if self.model is None:
            raise ValueError("Model has not been fit yet.")

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        """Return the pre-sigmoid logit ``w^T x + b`` for each example (shape ``(n,)``)."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        logits = self.model.decision_function(X)
        return np.asarray(logits, dtype=np.float64).reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities, shape ``(n, 2)``."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return np.asarray(self.model.predict_proba(X), dtype=np.float64)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Threshold ``P(y = classes_[1])`` at ``threshold`` (default 0.5, matching NN)."""
        probs_pos = self.predict_proba(X)[:, 1]
        pred_positive = probs_pos >= threshold
        return np.where(pred_positive, self.classes_[1], self.classes_[0]).astype(int)

    # ----------------------------------------------------- LR-specific

    def get_coefficients(self) -> tuple[np.ndarray, float]:
        """Return ``(w, b)`` for the interpretability discussion."""
        self._check_fitted()
        w = np.asarray(self.model.coef_, dtype=np.float64).reshape(-1)
        b = float(np.asarray(self.model.intercept_).reshape(-1)[0])
        return w, b

    def nonconformity_scores(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Split-conformal nonconformity score ``1 - P(y_true | x)``."""
        self._check_fitted()
        probs = self.predict_proba(X)
        y = np.asarray(y).ravel()
        class_idx = np.searchsorted(self.classes_, y)
        p_true = probs[np.arange(len(y)), class_idx]
        return 1.0 - p_true

    # ------------------------------------------------------------ utils

    def get_params(self) -> dict:
        return {
            "penalty": self.penalty,
            "C": self.C,
            "Cs": list(self.Cs),
            "cv": self.cv,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "scoring": self.scoring,
        }

    def clone(self) -> "LogisticRegressionClassifier":
        """Deep copy of the classifier (matches ``NeuralNetClassifier.clone()``)."""
        return copy.deepcopy(self)
