from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"))

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.experiment_pipeline import ExperimentConfig, run_full_experiment


def main() -> None:
    outputs = run_full_experiment(config=ExperimentConfig())
    for name, df in outputs.items():
        print(f"\n{name}")
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
