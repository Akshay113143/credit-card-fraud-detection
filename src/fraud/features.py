"""Feature construction. Stateless, so it cannot leak test statistics."""
import numpy as np
import pandas as pd

PCA_COLS = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_COLS + ["log_amount", "hour_sin", "hour_cos"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y).

    - log_amount: Amount is heavily right-skewed; log1p compresses it and keeps 0 valid.
    - hour_sin / hour_cos: Time is seconds since the first transaction. Raw Time only
      says "day 1 vs day 2", which does not generalise to future data, so it is dropped.
      Its position within a 24h cycle is kept, encoded on a circle so 23:59 sits next to 00:00.
    """
    X = df[PCA_COLS].copy()
    X["log_amount"] = np.log1p(df["Amount"])
    hour = (df["Time"] % 86400) / 3600.0
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return X[FEATURES], df["Class"].astype(int)
