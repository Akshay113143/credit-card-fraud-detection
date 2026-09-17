import numpy as np

from fraud import evaluate as E
from fraud import models as M
from fraud import threshold as T

y = np.array([1, 0, 1, 0, 0, 1])
p = np.array([0.9, 0.8, 0.6, 0.3, 0.2, 0.1])
amt = np.array([100.0, 50.0, 10.0, 5.0, 5.0, 1000.0])


def test_evaluate_flags_counts_and_cost():
    r = T.evaluate_flags(y, p >= 0.5, amt, review_cost=2.0)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2)
    assert r["review_cost_eur"] == 6.0            # 3 alerts x 2
    assert r["missed_fraud_eur"] == 1000.0         # the 0.1-scored fraud
    assert r["total_cost_eur"] == 1006.0
    assert np.isclose(r["saving_vs_no_model_pct"], 100 * (1 - 1006 / 1110))


def test_cost_optimal_threshold_matches_brute_force():
    rng = np.random.default_rng(3)
    yy = (rng.random(300) < 0.1).astype(int)
    pp = np.clip(rng.random(300) * 0.5 + yy * 0.4, 0, 1).round(3)   # rounded -> ties
    aa = rng.lognormal(3, 1, 300)
    t = T.cost_optimal_threshold(yy, pp, aa, 5.0)
    best = min(T.candidate_thresholds(pp),
               key=lambda th: T.evaluate_flags(yy, pp >= th, aa, 5.0)["total_cost_eur"])
    got = T.evaluate_flags(yy, pp >= t, aa, 5.0)["total_cost_eur"]
    want = T.evaluate_flags(yy, pp >= best, aa, 5.0)["total_cost_eur"]
    assert np.isclose(got, want)


def test_threshold_for_recall():
    t = T.threshold_for_recall(y, p, 0.6)          # need 2 of 3 frauds
    assert t == 0.6
    assert T.evaluate_flags(y, p >= t)["recall"] >= 0.6


def test_expected_cost_rule():
    flags = T.expected_cost_flags(np.array([0.1, 0.1, 0.9]), np.array([100, 10, 1]), 5.0)
    assert flags.tolist() == [True, False, False]
    guarded = T.guarded_flags(np.array([0.1, 0.1, 0.9]), np.array([100, 10, 1]), 5.0)
    assert guarded.tolist() == [True, False, True]


def test_precision_at_k():
    assert E.precision_at_k(y, p, 2) == 0.5
    assert E.precision_at_k(y, p, 1) == 1.0


def test_bootstrap_is_reproducible():
    a = E.bootstrap_pr_auc(y, {"m": p}, n=50, seed=1)["m"]["mean"]
    b = E.bootstrap_pr_auc(y, {"m": p}, n=50, seed=1)["m"]["mean"]
    assert a == b


def test_platt_is_monotonic_and_bounded():
    rng = np.random.default_rng(0)
    raw = rng.random(500); yy = (rng.random(500) < raw * 0.2).astype(int)
    cal = M.PlattCalibrator().fit(raw, yy)
    grid = np.linspace(0.001, 0.999, 50)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= 0)
    assert out.min() >= 0 and out.max() <= 1


def test_models_learn_toy_signal(toy_df):
    from fraud import features as F
    X, yy = F.build_features(toy_df)
    m = M.make_xgb({"n_estimators": 50, "max_depth": 3}, 19.0, 0).fit(X, yy)
    assert E.ranking_metrics(yy, m.predict_proba(X)[:, 1])["pr_auc"] > 0.8
