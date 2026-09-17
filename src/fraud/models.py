"""Model factories and probability calibration."""
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_logreg(seed: int) -> Pipeline:
    # Scaler lives inside the Pipeline, so it is fit on training rows only.
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def make_random_forest(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed,
    )


def make_xgb(params: dict, scale_pos_weight: float, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight, random_state=seed)


class PlattCalibrator:
    """Platt scaling: fit a 1-D logistic regression on the model's log-odds.

    scale_pos_weight inflates fraud scores on purpose, so raw outputs are not
    probabilities. Platt scaling is monotonic, so rankings (PR-AUC, ROC-AUC) do not change;
    only the numbers become usable as probabilities.
    """

    def __init__(self):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p_raw, y):
        self.lr.fit(self._logit(p_raw), y)
        return self

    def transform(self, p_raw):
        return self.lr.predict_proba(self._logit(p_raw))[:, 1]
