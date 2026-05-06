from __future__ import annotations

import numpy as np

from src.abstention.conformal import SplitConformalAbstainer
from src.abstention.learned import LearnedAbstainer, probability_abstention_features
from src.evaluation.calibration_metrics import brier_score, exp_cal_error
from src.evaluation.classification_metrics import binary_classification_summary
from src.evaluation.risk_coverage import risk_coverage_at_threshold


def test_calibration_metrics_accept_binary_and_matrix_probs() -> None:
    probs_pos = np.array([0.9, 0.7, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])
    probs = np.column_stack([1.0 - probs_pos, probs_pos])

    assert brier_score(probs_pos, y) == brier_score(probs, y)
    assert exp_cal_error(probs, y, n_bins=5) >= 0.0


def test_binary_classification_summary_reports_class_specific_error() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    summary = binary_classification_summary(y_true, y_pred)

    assert summary["accuracy"] == 0.75
    assert summary["malignant_recall"] == 0.5
    assert summary["benign_recall"] == 1.0
    assert summary["false_benign_rate"] == 0.5


def test_risk_coverage_threshold_retains_only_confident_predictions() -> None:
    probs = np.array([[0.9, 0.1], [0.55, 0.45], [0.2, 0.8]])
    y = np.array([0, 0, 1])
    out = risk_coverage_at_threshold(probs, y, threshold=0.8)

    assert out["coverage"] == 2 / 3
    assert out["risk"] == 0.0


def test_split_conformal_abstainer_abstains_on_non_singleton_sets() -> None:
    cal_probs = np.array([[0.95, 0.05], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    y_cal = np.array([0, 1, 0, 1])
    test_probs = np.array([[0.99, 0.01], [0.52, 0.48], [0.01, 0.99]])

    abstainer = SplitConformalAbstainer(alpha=0.25).fit(cal_probs, y_cal)
    preds, mask, sets = abstainer.apply(test_probs)

    assert sets.shape == test_probs.shape
    assert preds[0] == 0
    assert preds[2] == 1
    assert mask[1]


def test_learned_abstainer_scores_probability_features() -> None:
    cal_probs = np.array(
        [
            [0.95, 0.05],
            [0.90, 0.10],
            [0.45, 0.55],
            [0.40, 0.60],
            [0.20, 0.80],
            [0.10, 0.90],
        ]
    )
    y_cal = np.array([0, 0, 0, 1, 1, 1])
    test_probs = np.array([[0.99, 0.01], [0.51, 0.49], [0.05, 0.95]])

    features = probability_abstention_features(test_probs)
    assert features.shape == (3, 4)

    abstainer = LearnedAbstainer(target_coverage=0.7, random_state=0).fit(cal_probs, y_cal)
    preds, mask, scores = abstainer.apply(test_probs)

    assert preds.shape == (3,)
    assert mask.shape == (3,)
    assert scores.shape == (3,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
