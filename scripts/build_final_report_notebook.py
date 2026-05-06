from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "final_project_report.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    nb.cells = [
        md(
            """
            # Reliability Under Distribution Shift: Calibration and Abstention
            Project mentor: CS 475/675 course staff

            Team member: Justin Wang

            Repository: `ml_reliability_under_shift`
            """
        ),
        md(
            """
            # Outline and Deliverables

            ## Must Accomplish: Completed

            1. Train baseline classifiers for the Breast Cancer Wisconsin diagnostic task. We completed this with a cross-validated logistic regression model and a small feedforward neural network.
            2. Evaluate calibration and reliability on clean data. We completed this with log loss, Brier score, expected calibration error, reliability diagrams, and post-hoc temperature scaling.
            3. Evaluate behavior under distribution shift. We completed this with matched synthetic covariate shifts across multiple severities and perturbation replicates.

            ## Expect To Accomplish: Completed

            1. Apply abstention on top of calibrated probabilities. We completed this with calibrated threshold abstention and calibrated split conformal abstention.
            2. Compare reliability/abstention behavior across model classes. We completed this for both logistic regression and the neural network.
            3. Use class-sensitive metrics. We added malignant recall, benign recall, false-benign rate, balanced accuracy, and macro metrics.

            ## Would Like To Accomplish: Partially Completed

            1. Learned abstention model: completed as a lightweight learned correctness predictor over calibrated uncertainty features. Its shifted coverage is lower than its calibration target, so we treat it as an informative stretch result rather than the cleanest operating point.
            2. Real external-shift dataset: not completed. We instead used matched synthetic covariate shifts so clean-vs-shifted comparisons are controlled and reproducible.

            ## Additional Work

            We added class-sensitive evaluation for malignant and benign recall. This matters because overall accuracy can look high even when the model makes the most costly error: predicting benign for a malignant case.
            """
        ),
        md(
            """
            # Preliminaries

            We studied how binary classifiers behave when test-time covariates shift away from the clean training distribution. The concrete task is breast-cancer diagnosis from continuous tumor measurements. In this setting reliability is not just whether the top-1 prediction is correct; it is also whether the model's confidence remains meaningful when measurements change in a plausible way.

            This project connects to several course themes: proper losses and probabilistic prediction, calibration, distribution shift, model selection, and uncertainty-aware decision rules. The ethical stakes are also clear. A model that is confidently wrong under a new hospital, scanner, staining protocol, or measurement process could cause harm if used without human review. Abstention is one way to make the model less brittle: when uncertainty is high, the system can defer instead of forcing a diagnosis.
            """
        ),
        md(
            """
            ## Dataset

            We used the `sklearn.datasets.load_breast_cancer` dataset, a cleaned version of the Breast Cancer Wisconsin diagnostic data. It contains 569 examples, 30 continuous features computed from digitized cell nuclei, and a binary label where `0` is malignant and `1` is benign. The dataset is small but useful for this project because the features are interpretable and the task is sensitive to class-specific errors.
            """
        ),
        code(
            """
            import os
            import sys
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd
            from IPython.display import Image, display
            from sklearn.datasets import load_breast_cancer

            REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
            sys.path.insert(0, str(REPO_ROOT))

            ARTIFACT_DIR = REPO_ROOT / "artifacts" / "final_report"
            TABLE_DIR = ARTIFACT_DIR / "tables"
            FIGURE_DIR = ARTIFACT_DIR / "figures"

            data = load_breast_cancer()
            print(f"n_examples = {data.data.shape[0]}")
            print(f"n_features = {data.data.shape[1]}")
            print(dict(zip(data.target_names, pd.Series(data.target).value_counts().sort_index())))
            pd.DataFrame(data.data[:3], columns=data.feature_names)
            """
        ),
        md(
            """
            ## Pre-processing

            All features are continuous. We used a stratified 60/20/20 split into train, calibration, and test sets. The calibration split is held out from training so temperature scaling and conformal thresholds do not reuse training examples. We fit a `StandardScaler` only on the training split and applied it to calibration, clean test, and shifted test copies. This avoids leaking test-set statistics into the training pipeline.
            """
        ),
        code(
            """
            from src.evaluation.experiment_pipeline import prepare_breast_cancer_split

            split = prepare_breast_cancer_split(random_state=42)
            examples = pd.DataFrame(split["X_test_raw"][:3], columns=data.feature_names)
            scaled_examples = pd.DataFrame(split["X_test"][:3], columns=data.feature_names)
            display(examples)
            display(scaled_examples)
            """
        ),
        md(
            """
            # Models and Evaluation

            ## Experimental Setup

            The two base classifiers are logistic regression and a feedforward neural network. Logistic regression uses cross-validation over L2 regularization strength with log-loss scoring. The neural network is a compact multilayer perceptron trained with binary cross-entropy with logits.

            We evaluate clean and shifted performance with accuracy, balanced accuracy, malignant recall, benign recall, ROC AUC, log loss, Brier score, and expected calibration error. We report raw confidence behavior and temperature-scaled ECE. For abstention we use selective risk: the error rate among retained, non-abstained predictions. To match our proposal, the main abstention results are applied after temperature scaling, so the abstention rule sees calibrated probabilities rather than raw model scores.
            """
        ),
        code(
            """
            clean_metrics = pd.read_csv(TABLE_DIR / "clean_metrics.csv")
            shift_summary = pd.read_csv(TABLE_DIR / "shift_severity_summary.csv")
            abstention_summary = pd.read_csv(TABLE_DIR / "abstention_summary.csv")

            clean_metrics.round(4)
            """
        ),
        md(
            """
            ## Baselines

            Logistic regression is a strong baseline here because the dataset has only 30 standardized continuous features and the classes are close to linearly separable. The neural network baseline asks whether a more flexible non-linear classifier improves reliability, not just accuracy. On clean data, logistic regression performs best on accuracy and balanced accuracy, while the neural network has slightly lower clean ECE.
            """
        ),
        code(
            """
            display(clean_metrics[[
                "model", "accuracy", "balanced_accuracy", "malignant_recall",
                "benign_recall", "roc_auc", "ece", "temp_ece", "brier", "log_loss"
            ]].round(4))
            display(pd.read_csv(TABLE_DIR / "clean_confusion_matrices.csv"))
            """
        ),
        md(
            """
            ## Methods

            The shift generator creates matched shifted copies of the same clean test examples. That means any clean-vs-shifted difference is caused by perturbed features rather than by a different sample composition. The perturbations target features we identified as plausible sources of acquisition or population shift, especially malignant examples' texture and worst-case geometric measurements, plus small background jitter on other features.

            Temperature scaling is fit on calibration logits to test whether simple post-hoc calibration improves confidence estimates. Abstention is then evaluated in three forms on top of those calibrated probabilities: confidence thresholding at 0.8, split conformal prediction sets with alpha = 0.1 fit on calibrated calibration probabilities, and a learned abstainer trained on calibration-set uncertainty features to predict whether the base classifier will be correct. For transparency, the artifact table also includes raw-probability abstention variants, but the proposal-aligned method is calibrated + abstention. The conformal method abstains when its prediction set is empty or has more than one label.
            """
        ),
        code(
            """
            display(Image(filename=str(FIGURE_DIR / "clean_reliability.png")))
            display(Image(filename=str(FIGURE_DIR / "shift_sweep.png")))
            """
        ),
        md(
            """
            # Results

            The shift results do not show a dramatic accuracy collapse, which is an important finding rather than a failure. The test set is small and the shift is matched, so the decision boundary often remains correct. However, reliability metrics still move in the expected direction. For logistic regression, accuracy drops from 0.9825 clean to 0.9614 at severity 2.0, while Brier score increases from 0.0484 to 0.0524. For the neural net, accuracy remains roughly stable, but Brier score increases from 0.0508 to 0.0584 and log loss increases from 0.1033 to 0.1091 at severity 2.0.

            Temperature scaling helps on clean data and often under shift, but it is not a cure-all. This matches the core lesson: calibration is distribution-dependent. A scalar temperature learned on clean calibration examples cannot fully repair a shifted test distribution.
            """
        ),
        code(
            """
            display(shift_summary[[
                "model", "severity", "accuracy", "accuracy_std", "balanced_accuracy",
                "malignant_recall", "ece", "temp_ece", "brier", "log_loss",
                "delta_accuracy", "delta_brier", "delta_log_loss"
            ]].round(4))
            """
        ),
        md(
            """
            ## Abstention Results

            Abstention is the clearest robustness result, and it is now applied on top of calibration as intended in the proposal. At representative severity 1.5, logistic regression's un-abstained selective risk is 0.0351. Calibrated threshold abstention reduces that to 0.0093 while retaining 94.74% of examples. Calibrated split conformal abstention gives similar risk, 0.0099, while retaining 88.60%. The calibrated learned abstainer drives logistic selective risk to 0.0000, but with lower shifted coverage of 80.70%, so it is best interpreted as an aggressive stretch method rather than the cleanest operating point.

            For the neural network, calibrated conformal abstention reduces selective risk from 0.0263 to 0.0096 at 91.23% coverage. The calibrated learned abstainer reaches similar selective risk, 0.0105, at 83.33% coverage. This supports the project hypothesis: calibrated selective prediction can preserve reliability on retained cases, at the cost of coverage. It also shows a limitation of learned abstention under shift: a target coverage learned on calibration data may not transfer exactly to shifted data.
            """
        ),
        code(
            """
            calibrated_abstention = abstention_summary[
                abstention_summary["probability_source"].eq("calibrated")
            ]
            display(calibrated_abstention.round(4))
            display(abstention_summary.round(4))
            display(Image(filename=str(FIGURE_DIR / "risk_coverage.png")))
            display(Image(filename=str(FIGURE_DIR / "abstention_risk.png")))
            """
        ),
        md(
            """
            # Discussion

            ## What We Learned

            The biggest lesson is that high clean accuracy is not enough evidence of reliability. Logistic regression looks excellent in distribution, but even a modest matched shift changes its calibration and increases class-sensitive error. The neural network has slightly better clean ECE but worse clean accuracy, and its log loss worsens under stronger shifts. This is a useful reminder that different metrics answer different questions: accuracy measures decisions, ECE measures confidence alignment, and log loss punishes confident mistakes.

            We also learned that abstention should be evaluated with coverage and should consume calibrated uncertainty estimates when calibration is part of the method. A lower error rate is not meaningful if the model abstains on almost everything. In our experiments, calibrated threshold and conformal abstention reduced selective risk while keeping roughly 89-95% coverage, which is a practically meaningful tradeoff. Learned abstention reduced risk too, but its shifted coverage was lower than its calibration target, which is exactly the kind of failure mode that distribution-shift evaluation is meant to reveal.

            With two more weeks, the next step would be a larger or more realistic external-shift dataset. The synthetic matched shift is useful because it gives controlled comparisons, but a real hospital-to-hospital or scanner-to-scanner shift would be a stronger test of whether these reliability tools transfer beyond the toy setting.
            """
        ),
        md(
            """
            # Reproducibility

            The final artifacts in this notebook were generated with:

            ```bash
            python scripts/run_final_experiment.py
            python scripts/build_final_report_notebook.py
            python -m pytest tests/test_metrics_and_abstention.py
            ```

            The reusable implementation lives in `src/evaluation/experiment_pipeline.py`, `src/evaluation/classification_metrics.py`, `src/abstention/conformal.py`, and `src/abstention/learned.py`.
            """
        ),
        md(
            """
            # References

            - Wolberg, W. H., Street, W. N., and Mangasarian, O. L. Breast Cancer Wisconsin diagnostic dataset.
            - Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks.
            - Geifman, Y. and El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.
            - Vovk, V., Gammerman, A., and Shafer, G. (2005). Algorithmic Learning in a Random World.
            """
        ),
    ]

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
