"""Project-wide settings. Every script and the notebook read from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "creditcard.csv"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"

SEED = 42

# Chronological split: first 60% of transactions (by Time) train, next 20% validate, last 20% test.
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

# Cost model (EUR). Every alert is reviewed by an analyst; a missed fraud loses its amount.
REVIEW_COST = 5.0
REVIEW_COST_GRID = [1.0, 2.0, 5.0, 10.0, 20.0]

# Small hyperparameter grid, scored on validation PR-AUC.
XGB_BASE = dict(
    n_estimators=400,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
)
XGB_GRID = [
    {"max_depth": d, "min_child_weight": m}
    for d in (3, 5, 7)
    for m in (1, 5)
]

N_BOOTSTRAP = 1000
ALERT_BUDGET = 100          # "top-k alerts" metric
SHAP_SAMPLE = 2000
