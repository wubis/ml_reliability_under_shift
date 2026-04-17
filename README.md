# ML Reliability Under Shift: Abstention Study
Study of how uncertainty estimates degrade under distribution shift and how abstention-based methods improve robustness and reliability.

Below is a draft of what the final repository might look like
```
ml_reliability_under_shift/
│
├── notebooks/
│   ├── 01_data.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_calibration.ipynb
│   ├── 04_abstention.ipynb
│   └── 05_shift_evaluation.ipynb
│
├── src/
│   ├── models/
│   │   ├── baseline_logistic.py
│   │   └── baseline_nn.py
│   │
│   ├── calibration/
│   │   ├── base.py
│   │   └── temperature.py
│   │
│   ├── abstention/
│   │   ├── base.py
│   │   ├── threshold.py
│   │   ├── conformal.py
│   │   └── learned.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── calibration_metrics.py
│   │   └── risk_coverage.py
│   │
│   └── docs/
│       └── tbd
│
└── requirements.txt
│
└── README.md
```
