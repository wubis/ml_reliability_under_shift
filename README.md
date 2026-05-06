# ML Reliability Under Shift: Abstention Study

Study of how uncertainty estimates degrade under distribution shift and how abstention-based methods improve robustness and reliability on Scikit-Learn's Breast Cancer Wisconsin diagnostic dataset.

## Final Report

The final writeup is in `notebooks/final_project_report.ipynb`.

## Reproducing Results

```bash
python scripts/run_final_experiment.py
python scripts/build_final_report_notebook.py
python -m pytest tests/test_metrics_and_abstention.py
```

The experiment writes tables and figures to `artifacts/final_report/`.

## What Is Implemented

- Logistic regression and feedforward neural-network baselines.
- Calibration metrics: log loss, Brier score, expected calibration error, and reliability diagrams.
- Temperature scaling fit on a held-out calibration split.
- Matched synthetic covariate shifts over breast-cancer features at multiple severities.
- Class-sensitive evaluation, including malignant recall and false-benign rate.
- Confidence-threshold, split-conformal, and learned abstention applied to calibrated probabilities.
- Risk-coverage and selective-risk evaluation under representative shift.
