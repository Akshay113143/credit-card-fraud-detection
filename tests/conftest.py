import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def toy_df():
    """Small frame with the real dataset's columns; no download needed."""
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = rng.lognormal(3, 1, n).round(2)
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    df.loc[df["Class"] == 1, "V14"] -= 4          # learnable signal
    return df
