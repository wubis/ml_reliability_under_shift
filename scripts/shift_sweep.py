from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.load_shifts import sweep_matched_shift_severities
from src.evaluation.shift_experiment import run_shift_severity_sweep
from src.models.baseline_logistic import LogisticRegressionClassifier
from src.models.baseline_nn import NeuralNetClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)

    severities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    num_replicates = 7

    models = {
        "logistic": LogisticRegressionClassifier(random_state=42),
        "nn": NeuralNetClassifier(hidden_dims=[32, 16], epochs=100, random_state=42),
    }

    all_rows = []
    for model_name, model in models.items():
        model.fit(X_train_s, y_train)
        clean_row = None
        shifted_replicates = []
        for replicate in range(num_replicates):
            shifted_raw = sweep_matched_shift_severities(
                X_test,
                y_test,
                severities=severities,
                random_state=42 + 100 * replicate,
            )
            shifted_sets = {severity: scaler.transform(X_shift) for severity, X_shift in shifted_raw.items()}
            rows = run_shift_severity_sweep(
                model=model,
                X_cal=X_cal_s,
                y_cal=y_cal,
                X_clean=X_test_s,
                y_clean=y_test,
                shifted_sets=shifted_sets,
                abstain_threshold=0.8,
            )
            if clean_row is None:
                clean_row = rows[0]
            for row in rows[1:]:
                row["replicate"] = replicate
                shifted_replicates.append(row)

        assert clean_row is not None
        clean_row["model"] = model_name
        clean_row["replicate"] = -1
        all_rows.append(clean_row)

        shifted_df = pd.DataFrame(shifted_replicates)
        averaged = shifted_df.groupby(["split", "severity"], as_index=False).mean(numeric_only=True)
        for row in averaged.to_dict(orient="records"):
            row["model"] = model_name
            row["replicate"] = num_replicates
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    cols = [
        "model",
        "split",
        "severity",
        "accuracy",
        "ece",
        "temp_ece",
        "brier",
        "log_loss",
        "delta_accuracy",
        "delta_ece",
        "delta_brier",
        "delta_log_loss",
        "rc_coverage",
        "rc_risk",
        "rc_accuracy",
        "rc_abstention_rate",
    ]
    print(df[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
