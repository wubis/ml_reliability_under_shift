from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.abstention.conformal import SplitConformalAbstainer
from src.abstention.learned import LearnedAbstainer
from src.abstention.threshold import ThresholdAbstainer
from src.calibration.temperature import TemperatureScaler
from src.evaluation.classification_metrics import binary_classification_summary
from src.evaluation.load_shifts import sweep_matched_shift_severities
from src.evaluation.risk_coverage import risk_coverage_curve
from src.evaluation.shift_evaluation import summarize_predictions
from src.models.baseline_logistic import LogisticRegressionClassifier
from src.models.baseline_nn import NeuralNetClassifier


@dataclass(frozen=True)
class ExperimentConfig:
    random_state: int = 42
    severities: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    num_replicates: int = 5
    abstain_threshold: float = 0.8
    learned_target_coverage: float = 0.9
    representative_severity: float = 1.5
    n_bins: int = 10


def prepare_breast_cancer_split(random_state: int = 42) -> dict[str, object]:
    data = load_breast_cancer()
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=random_state,
        stratify=data.target,
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.25,
        random_state=random_state,
        stratify=y_trainval,
    )
    scaler = StandardScaler().fit(X_train)
    return {
        "data": data,
        "scaler": scaler,
        "X_train_raw": X_train,
        "X_cal_raw": X_cal,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_cal": y_cal,
        "y_test": y_test,
        "X_train": scaler.transform(X_train),
        "X_cal": scaler.transform(X_cal),
        "X_test": scaler.transform(X_test),
    }


def default_models(random_state: int = 42) -> dict[str, object]:
    return {
        "logistic": LogisticRegressionClassifier(random_state=random_state),
        "neural_net": NeuralNetClassifier(hidden_dims=[32, 16], epochs=100, random_state=random_state),
    }


def run_full_experiment(
    output_dir: str | Path = "artifacts/final_report",
    config: ExperimentConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or ExperimentConfig()
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    split = prepare_breast_cancer_split(config.random_state)
    models = default_models(config.random_state)

    fitted: dict[str, object] = {}
    clean_rows: list[dict[str, float | str]] = []
    shift_rows: list[dict[str, float | str]] = []
    abstention_rows: list[dict[str, float | str]] = []
    cached_clean_probs: dict[str, np.ndarray] = {}
    cached_shift_probs: dict[str, np.ndarray] = {}

    X_train = split["X_train"]
    X_cal = split["X_cal"]
    X_test = split["X_test"]
    X_test_raw = split["X_test_raw"]
    y_train = split["y_train"]
    y_cal = split["y_cal"]
    y_test = split["y_test"]
    scaler = split["scaler"]

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        fitted[model_name] = model

        clean_probs = model.predict_proba(X_test)
        clean_logits = model.predict_logits(X_test)
        cal_probs = model.predict_proba(X_cal)
        cal_logits = model.predict_logits(X_cal)
        temp_scaler = TemperatureScaler().fit(cal_logits, y_cal)
        clean_temp_probs = temp_scaler.predict_proba(clean_logits)
        clean_preds = clean_probs.argmax(axis=1)
        cached_clean_probs[model_name] = clean_probs

        clean_rows.append(
            {
                "model": model_name,
                **binary_classification_summary(y_test, clean_preds, clean_probs),
                **summarize_predictions(clean_probs, y_test, n_bins=config.n_bins),
                "log_loss": float(log_loss(y_test, clean_probs)),
                "temp_ece": summarize_predictions(clean_temp_probs, y_test, n_bins=config.n_bins)["ece"],
            }
        )

        for replicate in range(config.num_replicates):
            shifted_raw = sweep_matched_shift_severities(
                X_test_raw,
                y_test,
                severities=np.array(config.severities),
                random_state=config.random_state + 100 * replicate,
            )
            for severity, X_shift_raw in shifted_raw.items():
                X_shift = scaler.transform(X_shift_raw)
                probs = model.predict_proba(X_shift)
                logits = model.predict_logits(X_shift)
                temp_probs = temp_scaler.predict_proba(logits)
                preds = probs.argmax(axis=1)
                row = {
                    "model": model_name,
                    "split": "shifted",
                    "severity": float(severity),
                    "replicate": float(replicate),
                    **binary_classification_summary(y_test, preds, probs),
                    **summarize_predictions(probs, y_test, n_bins=config.n_bins),
                    "log_loss": float(log_loss(y_test, probs)),
                    "temp_ece": summarize_predictions(temp_probs, y_test, n_bins=config.n_bins)["ece"],
                }
                shift_rows.append(row)

        representative_shift_raw = sweep_matched_shift_severities(
            X_test_raw,
            y_test,
            severities=np.array([config.representative_severity]),
            random_state=config.random_state,
        )[config.representative_severity]
        representative_shift = scaler.transform(representative_shift_raw)
        shift_probs = model.predict_proba(representative_shift)
        shift_logits = model.predict_logits(representative_shift)
        shift_temp_probs = temp_scaler.predict_proba(shift_logits)
        cached_shift_probs[model_name] = shift_temp_probs

        threshold_abstainer = ThresholdAbstainer(threshold=config.abstain_threshold)
        cal_temp_probs = temp_scaler.predict_proba(cal_logits)
        raw_conformal_abstainer = SplitConformalAbstainer(alpha=0.1).fit(cal_probs, y_cal)
        calibrated_conformal_abstainer = SplitConformalAbstainer(alpha=0.1).fit(cal_temp_probs, y_cal)
        raw_learned_abstainer = LearnedAbstainer(
            target_coverage=config.learned_target_coverage,
            random_state=config.random_state,
        ).fit(cal_probs, y_cal)
        calibrated_learned_abstainer = LearnedAbstainer(
            target_coverage=config.learned_target_coverage,
            random_state=config.random_state,
        ).fit(cal_temp_probs, y_cal)

        abstention_configs = (
            ("raw", "none", shift_probs, np.zeros_like(y_test, dtype=bool)),
            ("calibrated", "none", shift_temp_probs, np.zeros_like(y_test, dtype=bool)),
            (
                "raw",
                f"threshold_{config.abstain_threshold}",
                shift_probs,
                threshold_abstainer.abstain_mask(shift_probs),
            ),
            (
                "calibrated",
                f"threshold_{config.abstain_threshold}",
                shift_temp_probs,
                threshold_abstainer.abstain_mask(shift_temp_probs),
            ),
            (
                "raw",
                "split_conformal_alpha_0.1",
                shift_probs,
                raw_conformal_abstainer.abstain_mask(shift_probs),
            ),
            (
                "calibrated",
                "split_conformal_alpha_0.1",
                shift_temp_probs,
                calibrated_conformal_abstainer.abstain_mask(shift_temp_probs),
            ),
            (
                "raw",
                f"learned_target_coverage_{config.learned_target_coverage}",
                shift_probs,
                raw_learned_abstainer.abstain_mask(shift_probs),
            ),
            (
                "calibrated",
                f"learned_target_coverage_{config.learned_target_coverage}",
                shift_temp_probs,
                calibrated_learned_abstainer.abstain_mask(shift_temp_probs),
            ),
        )
        for probability_source, method_name, method_probs, mask in abstention_configs:
            method_preds = method_probs.argmax(axis=1)
            keep = ~mask
            retained_summary = binary_classification_summary(y_test[keep], method_preds[keep]) if np.any(keep) else {}
            abstention_rows.append(
                {
                    "model": model_name,
                    "probability_source": probability_source,
                    "method": method_name,
                    "severity": config.representative_severity,
                    "coverage": float(keep.mean()),
                    "abstention_rate": float(mask.mean()),
                    "selective_risk": float((method_preds[keep] != y_test[keep]).mean()) if np.any(keep) else 0.0,
                    "retained_count": float(keep.sum()),
                    "malignant_recall_retained": retained_summary.get("malignant_recall", 0.0),
                    "benign_recall_retained": retained_summary.get("benign_recall", 0.0),
                }
            )

    clean_df = pd.DataFrame(clean_rows)
    shift_df = pd.DataFrame(shift_rows)
    clean_baselines = clean_df.set_index("model")

    grouped_shift = (
        shift_df.groupby(["model", "severity"], as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            malignant_recall=("malignant_recall", "mean"),
            benign_recall=("benign_recall", "mean"),
            ece=("ece", "mean"),
            ece_std=("ece", "std"),
            temp_ece=("temp_ece", "mean"),
            brier=("brier", "mean"),
            log_loss=("log_loss", "mean"),
        )
    )
    for metric in ("accuracy", "balanced_accuracy", "malignant_recall", "ece", "brier", "log_loss"):
        grouped_shift[f"delta_{metric}"] = grouped_shift.apply(
            lambda row: float(row[metric] - clean_baselines.loc[row["model"], metric]),
            axis=1,
        )

    abstention_df = pd.DataFrame(abstention_rows)

    clean_df.to_csv(tables_dir / "clean_metrics.csv", index=False)
    grouped_shift.to_csv(tables_dir / "shift_severity_summary.csv", index=False)
    abstention_df.to_csv(tables_dir / "abstention_summary.csv", index=False)
    _write_figures(clean_df, grouped_shift, abstention_df, cached_clean_probs, cached_shift_probs, y_test, figures_dir)

    return {
        "clean_metrics": clean_df,
        "shift_severity_summary": grouped_shift,
        "abstention_summary": abstention_df,
    }


def _write_figures(
    clean_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    abstention_df: pd.DataFrame,
    clean_probs: dict[str, np.ndarray],
    shift_probs: dict[str, np.ndarray],
    y_test: np.ndarray,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for model_name in shift_df["model"].unique():
        subset = shift_df[shift_df["model"] == model_name]
        axes[0].plot(subset["severity"], subset["ece"], marker="o", label=model_name)
        axes[1].plot(subset["severity"], subset["accuracy"], marker="o", label=model_name)
    axes[0].set_xlabel("Shift severity")
    axes[0].set_ylabel("ECE")
    axes[0].set_title("Calibration degrades under shift")
    axes[1].set_xlabel("Shift severity")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Discrimination under shift")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "shift_sweep.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model_name, probs in clean_probs.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs[:, 1], n_bins=10, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Mean predicted P(benign)")
    ax.set_ylabel("Empirical benign fraction")
    ax.set_title("Clean reliability diagram")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "clean_reliability.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model_name, probs in shift_probs.items():
        curve = pd.DataFrame(risk_coverage_curve(probs, y_test, thresholds=np.linspace(0.0, 1.0, 51)))
        ax.plot(curve["coverage"], curve["risk"], marker=".", label=model_name)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk")
    ax.set_title("Calibrated risk-coverage at representative shift")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "risk_coverage.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    plot_df = abstention_df.copy()
    plot_df["variant"] = plot_df["probability_source"] + " + " + plot_df["method"]
    pivot = plot_df.pivot(index="model", columns="variant", values="selective_risk")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Selective risk")
    ax.set_title("Abstention reduces retained error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "abstention_risk.png", dpi=200)
    plt.close(fig)

    # Keep confusion matrices as data-bearing artifacts for the report text.
    rows = []
    for model_name, probs in clean_probs.items():
        cm = confusion_matrix(y_test, probs.argmax(axis=1), labels=[0, 1])
        rows.append({"model": model_name, "tn_malignant": cm[0, 0], "false_benign": cm[0, 1], "false_malignant": cm[1, 0], "tp_benign": cm[1, 1]})
    pd.DataFrame(rows).to_csv(figures_dir.parent / "tables" / "clean_confusion_matrices.csv", index=False)
