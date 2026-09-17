import numpy as np
import pytest

from fraud import data as D
from fraud import features as F


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        D.load_data(tmp_path / "nope.csv")


def test_load_data_rejects_wrong_columns(tmp_path, toy_df):
    path = tmp_path / "bad.csv"
    toy_df.drop(columns=["V3"]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        D.load_data(path)


def test_drop_exact_duplicates(toy_df):
    doubled = __import__("pandas").concat([toy_df, toy_df.head(10)])
    out, n = D.drop_exact_duplicates(doubled)
    assert n == 10 and len(out) == len(toy_df)


def test_chronological_split_is_ordered_and_disjoint(toy_df):
    shuffled = toy_df.sample(frac=1, random_state=1)
    tr, va, te = D.chronological_split(shuffled, 0.6, 0.2)
    assert len(tr) + len(va) + len(te) == len(toy_df)
    assert tr["Time"].max() <= va["Time"].min()
    assert va["Time"].max() <= te["Time"].min()


def test_features_shape_and_values(toy_df):
    X, y = F.build_features(toy_df)
    assert list(X.columns) == F.FEATURES
    assert "Time" not in X.columns and "Amount" not in X.columns
    assert not X.isna().any().any()
    np.testing.assert_allclose(X["hour_sin"] ** 2 + X["hour_cos"] ** 2, 1.0)
    assert (X["log_amount"] >= 0).all()
    assert y.sum() == toy_df["Class"].sum()


def test_hour_encoding_wraps_midnight():
    import pandas as pd
    base = {f"V{i}": [0.0, 0.0] for i in range(1, 29)}
    df = pd.DataFrame({"Time": [86399.0, 86400.0], **base, "Amount": [1.0, 1.0], "Class": [0, 0]})
    X, _ = F.build_features(df)
    assert abs(X["hour_sin"].iloc[0] - X["hour_sin"].iloc[1]) < 1e-3
