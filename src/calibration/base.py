from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseCalibrator(ABC):
    """
    Minimal shared interface for post-hoc probability calibrators.
    """

    @abstractmethod
    def fit(self, logits: np.ndarray, y: np.ndarray) -> "BaseCalibrator":
        """Estimate calibration parameters on a held-out calibration set."""

    @abstractmethod
    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        """Apply the learned calibration transform in logit space."""

    @abstractmethod
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Convert raw logits into calibrated probabilities."""
