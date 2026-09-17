"""Run the full pipeline: python scripts/run_pipeline.py"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud.pipeline import run_all  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    r = run_all(save=True)
    print(json.dumps({k: r[k] for k in ("data", "xgb_best_params", "model_comparison", "random_split",
                                        "calibration", "thresholds", "test_policies", "val_policies")}, indent=2))
    print(f"Done in {time.time() - t0:.0f}s. Figures in figures/, numbers in results/.")
