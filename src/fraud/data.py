"""Loading, validating and splitting the ULB credit card dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV and fail loudly if it is missing or not the expected dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud into data/."
        )
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)[:5]}...")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values.")
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows that are identical in every column (they would inflate test scores)."""
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Sort by Time and cut into train / validation / test blocks.

    The model is always evaluated on transactions that happen *after* everything it
    was trained on, which is how a deployed fraud model is actually used.
    """
    df = df.sort_values("Time", kind="mergesort").reset_index(drop=True)
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def random_split(df: pd.DataFrame, test_frac: float, seed: int):
    """Stratified random split, used only to show how optimistic it is."""
    return train_test_split(df, test_size=test_frac, random_state=seed, stratify=df["Class"])


def describe_split(name: str, part: pd.DataFrame) -> dict:
    return {
        "split": name,
        "rows": int(len(part)),
        "frauds": int(part["Class"].sum()),
        "fraud_rate_pct": round(float(part["Class"].mean() * 100), 4),
        "time_start_h": round(float(part["Time"].min() / 3600), 2),
        "time_end_h": round(float(part["Time"].max() / 3600), 2),
    }
