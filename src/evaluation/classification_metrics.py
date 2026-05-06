from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Class-sensitive summary for the Breast Cancer binary task.

    The sklearn dataset encodes malignant as 0 and benign as 1. Accuracy alone can
    hide clinically important failures, so this helper reports recall for both
    classes plus macro scores.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    malignant_total = cm[0].sum()
    benign_total = cm[1].sum()

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "malignant_recall": float(cm[0, 0] / malignant_total) if malignant_total else 0.0,
        "benign_recall": float(cm[1, 1] / benign_total) if benign_total else 0.0,
        "false_benign_rate": float(cm[0, 1] / malignant_total) if malignant_total else 0.0,
        "false_malignant_rate": float(cm[1, 0] / benign_total) if benign_total else 0.0,
    }

    if probs is not None:
        probs = np.asarray(probs, dtype=np.float64)
        if probs.ndim != 2 or probs.shape[0] != len(y_true) or probs.shape[1] != 2:
            raise ValueError("probs must have shape (n_samples, 2).")
        out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))

    return out
