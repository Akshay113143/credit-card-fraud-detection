"""End-to-end pipeline, split into steps so the notebook can show each one.

run_all() executes every step, writes figures/ and results/, and returns a dict.
"""
import json

import numpy as np
import pandas as pd
import shap

from . import config as C
from . import data as D
from . import evaluate as E
from . import features as F
from . import models as M
from . import plots as P
from . import threshold as T


def step_load():
    raw = D.load_data(C.DATA_PATH)
    df, n_dup = D.drop_exact_duplicates(raw)
    info = {
        "raw_rows": len(raw), "raw_frauds": int(raw["Class"].sum()),
        "duplicates_removed": n_dup,
        "duplicate_frauds_removed": int(raw["Class"].sum() - df["Class"].sum()),
        "rows": len(df), "frauds": int(df["Class"].sum()),
        "fraud_rate_pct": float(df["Class"].mean() * 100),
        "hours_covered": float(df["Time"].max() / 3600),
        "fraud_amount_median": float(df.loc[df.Class == 1, "Amount"].median()),
        "legit_amount_median": float(df.loc[df.Class == 0, "Amount"].median()),
    }
    return raw, df, info


def step_split(df):
    tr, va, te = D.chronological_split(df, C.TRAIN_FRAC, C.VAL_FRAC)
    table = pd.DataFrame([D.describe_split(n, p) for n, p in [("train", tr), ("validation", va), ("test", te)]])
    return tr, va, te, table


def step_features(tr, va, te):
    return F.build_features(tr), F.build_features(va), F.build_features(te)


def step_tune_xgb(Xtr, ytr, Xva, yva):
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    rows = []
    for g in C.XGB_GRID:
        m = M.make_xgb({**C.XGB_BASE, **g}, spw, C.SEED).fit(Xtr, ytr)
        rows.append({**g, "val_pr_auc": E.ranking_metrics(yva, m.predict_proba(Xva)[:, 1])["pr_auc"]})
    grid = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best = {k: int(grid.loc[0, k]) for k in ("max_depth", "min_child_weight")}
    model = M.make_xgb({**C.XGB_BASE, **best}, spw, C.SEED).fit(Xtr, ytr)
    return model, grid, best, spw


def step_models(Xtr, ytr, Xva, yva, Xte, yte, xgb_model):
    lr = M.make_logreg(C.SEED).fit(Xtr, ytr)
    rf = M.make_random_forest(C.SEED).fit(Xtr, ytr)
    fitted = {"Logistic regression": lr, "Random forest": rf, "XGBoost": xgb_model}
    val_p = {k: m.predict_proba(Xva)[:, 1] for k, m in fitted.items()}
    test_p = {k: m.predict_proba(Xte)[:, 1] for k, m in fitted.items()}
    rows = []
    for k in fitted:
        v = E.ranking_metrics(yva, val_p[k]); t = E.ranking_metrics(yte, test_p[k])
        tp, fp, fn, tn = T.confusion_at(yte, test_p[k], 0.5)
        rows.append({
            "model": k, "val_pr_auc": v["pr_auc"], "test_pr_auc": t["pr_auc"], "test_roc_auc": t["roc_auc"],
            "test_precision_at_100": E.precision_at_k(yte, test_p[k], C.ALERT_BUDGET),
            "recall_at_0.5": tp / (tp + fn), "precision_at_0.5": tp / (tp + fp) if tp + fp else 0.0,
            "tp_at_0.5": tp, "fp_at_0.5": fp,
        })
    return fitted, val_p, test_p, pd.DataFrame(rows)


def step_random_split_gap(df, best):
    """Same features and model, but a random stratified 80/20 split."""
    tr, te = D.random_split(df, 0.2, C.SEED)
    (Xtr, ytr), (Xte, yte) = F.build_features(tr), F.build_features(te)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    m = M.make_xgb({**C.XGB_BASE, **best}, spw, C.SEED).fit(Xtr, ytr)
    return E.ranking_metrics(yte, m.predict_proba(Xte)[:, 1])["pr_auc"], int(yte.sum())


def step_calibrate(p_val_raw, yva, p_test_raw, yte):
    cal = M.PlattCalibrator().fit(p_val_raw, yva)
    p_val, p_test = cal.transform(p_val_raw), cal.transform(p_test_raw)
    return cal, p_val, p_test, {
        "brier_raw": E.ranking_metrics(yte, p_test_raw)["brier"],
        "brier_calibrated": E.ranking_metrics(yte, p_test)["brier"],
        "pr_auc_raw": E.ranking_metrics(yte, p_test_raw)["pr_auc"],
        "pr_auc_calibrated": E.ranking_metrics(yte, p_test)["pr_auc"],
        "mean_raw_score_test": float(np.mean(p_test_raw)),
        "mean_calibrated_test": float(np.mean(p_test)),
        "test_fraud_rate": float(np.mean(yte)),
    }


def step_policies(p_val, yva, amt_val, p_test_raw, p_test, yte, amt_test, calibrator):
    """Choose every tunable number on VALIDATION, then score each policy once on test."""
    rc = C.REVIEW_COST
    t_cost = T.cost_optimal_threshold(yva, p_val, amt_val, rc)
    t_r90 = T.threshold_for_recall(yva, p_val, 0.90)
    thresholds = {"cost_optimal_global": t_cost, "recall90_global": t_r90,
                  "raw_0.5_in_calibrated_units": float(calibrator.transform([0.5])[0])}
    policies = {
        "Default 0.5": p_test_raw >= 0.5,
        "90% recall target": p_test >= t_r90,
        "Cost-optimal threshold": p_test >= t_cost,
        "Expected-cost rule": T.expected_cost_flags(p_test, amt_test, rc),
        "Expected-cost + guardrail": T.guarded_flags(p_test, amt_test, rc),
    }
    test_rows = {k: T.evaluate_flags(yte, f, amt_test, rc) for k, f in policies.items()}
    val_policies = {
        "Cost-optimal threshold": p_val >= t_cost,
        "Expected-cost rule": T.expected_cost_flags(p_val, amt_val, rc),
        "Expected-cost + guardrail": T.guarded_flags(p_val, amt_val, rc),
    }
    val_rows = {k: T.evaluate_flags(yva, f, amt_val, rc) for k, f in val_policies.items()}
    grid = np.unique(np.concatenate([np.geomspace(1e-6, 0.999, 400), [t_cost, t_r90]]))
    val_sweep = T.sweep(yva, p_val, grid, amt_val, rc)

    sens = []
    for c in C.REVIEW_COST_GRID:
        tg = T.cost_optimal_threshold(yva, p_val, amt_val, c)
        g = T.evaluate_flags(yte, p_test >= tg, amt_test, c)
        e = T.evaluate_flags(yte, T.expected_cost_flags(p_test, amt_test, c), amt_test, c)
        h = T.evaluate_flags(yte, T.guarded_flags(p_test, amt_test, c), amt_test, c)
        sens.append({"review_cost_eur": c, "global_threshold": tg,
                     "global_saving_pct": g["saving_vs_no_model_pct"],
                     "expected_cost_saving_pct": e["saving_vs_no_model_pct"], "expected_cost_alerts": e["alerts"],
                     "guardrail_saving_pct": h["saving_vs_no_model_pct"], "guardrail_alerts": h["alerts"],
                     "guardrail_recall": h["recall"]})
    return thresholds, test_rows, val_rows, val_sweep, pd.DataFrame(sens)


def step_test_sweep(p_test, yte, amt_test):
    ts = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    return T.sweep(yte, p_test, ts, amt_test, C.REVIEW_COST)


def step_shap(model, Xte, yte, p_test, flags):
    explainer = shap.TreeExplainer(model)
    sample = Xte.sample(min(C.SHAP_SAMPLE, len(Xte)), random_state=C.SEED)
    sv = explainer.shap_values(sample)
    importance = (pd.Series(np.abs(sv).mean(0), index=sample.columns)
                  .sort_values(ascending=False))

    def explain_row(pos):
        row = Xte.iloc[[pos]]
        vals = explainer.shap_values(row)[0]
        exp = shap.Explanation(values=vals, base_values=float(explainer.expected_value),
                               data=row.iloc[0].values, feature_names=list(Xte.columns))
        top = pd.Series(vals, index=Xte.columns).sort_values(key=np.abs, ascending=False).head(3)
        return exp, {
            "score_calibrated": float(p_test[pos]),
            "raw_log_odds": float(explainer.expected_value + vals.sum()),
            "base_log_odds": float(explainer.expected_value),
            "top_features": [{"feature": f, "value": float(row.iloc[0][f]), "shap": float(s)} for f, s in top.items()],
        }

    y = np.asarray(yte)
    fraud_pos = np.where(y == 1)[0]
    caught = fraud_pos[np.argmax(p_test[fraud_pos])]
    missed_candidates = fraud_pos[~np.asarray(flags)[fraud_pos]]
    missed = missed_candidates[np.argmin(p_test[missed_candidates])] if len(missed_candidates) else None
    exp_c, info_c = explain_row(int(caught))
    out = {"importance": importance, "shap_values": sv, "sample": sample,
           "caught": (exp_c, info_c), "missed": None}
    if missed is not None:
        out["missed"] = explain_row(int(missed))
    return out


def run_all(save=True):
    C.FIG_DIR.mkdir(exist_ok=True); C.RESULTS_DIR.mkdir(exist_ok=True)
    raw, df, data_info = step_load()
    tr, va, te, split_table = step_split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = step_features(tr, va, te)
    xgb_model, grid, best, spw = step_tune_xgb(Xtr, ytr, Xva, yva)
    fitted, val_p, test_p, comparison = step_models(Xtr, ytr, Xva, yva, Xte, yte, xgb_model)
    boot = E.bootstrap_pr_auc(yte, test_p, C.N_BOOTSTRAP, C.SEED)
    rand_pr, rand_frauds = step_random_split_gap(df, best)
    cal, p_val, p_test, cal_info = step_calibrate(val_p["XGBoost"], yva, test_p["XGBoost"], yte)
    thresholds, test_rows, val_rows, val_sweep, sens = step_policies(
        p_val, yva, va["Amount"].values, test_p["XGBoost"], p_test, yte, te["Amount"].values, cal)
    test_sweep = step_test_sweep(p_test, yte, te["Amount"].values)
    sh = step_shap(xgb_model, Xte, yte, p_test, T.guarded_flags(p_test, te["Amount"].values, C.REVIEW_COST))

    xgb_samples = boot["XGBoost"]["samples"]
    diffs = {k: xgb_samples - boot[k]["samples"] for k in boot if k != "XGBoost"}

    results = {
        "data": data_info,
        "splits": split_table.to_dict(orient="records"),
        "scale_pos_weight": spw,
        "xgb_grid": grid.to_dict(orient="records"),
        "xgb_best_params": best,
        "model_comparison": comparison.to_dict(orient="records"),
        "bootstrap_pr_auc": {k: {kk: vv for kk, vv in v.items() if kk != "samples"} for k, v in boot.items()},
        "bootstrap_diff_xgb_minus": {k: {"mean": float(d.mean()), "ci_low": float(np.percentile(d, 2.5)),
                                         "ci_high": float(np.percentile(d, 97.5))} for k, d in diffs.items()},
        "random_split": {"xgb_pr_auc": rand_pr, "test_frauds": rand_frauds,
                         "chronological_xgb_pr_auc": float(comparison.set_index("model").loc["XGBoost", "test_pr_auc"])},
        "calibration": cal_info,
        "review_cost_eur": C.REVIEW_COST,
        "thresholds": thresholds,
        "test_policies": test_rows,
        "val_policies": val_rows,
        "cost_sensitivity": sens.to_dict(orient="records"),
        "test_sweep": test_sweep.to_dict(orient="records"),
        "shap_top10": {k: float(v) for k, v in sh["importance"].head(10).items()},
        "shap_caught_example": sh["caught"][1],
        "shap_missed_example": sh["missed"][1] if sh["missed"] else None,
    }

    if save:
        import matplotlib.pyplot as plt
        fd = C.FIG_DIR
        P.class_balance(df, fd / "01_class_balance.png")
        P.amount_and_time(df, fd / "02_amount_time.png")
        cuts = [tr["Time"].max() / 3600, va["Time"].max() / 3600]
        P.split_timeline(df.sort_values("Time"), cuts, fd / "03_split_timeline.png")
        P.pr_roc(yte, test_p, fd / "04_pr_roc.png")
        P.bootstrap_hist(boot, fd / "05_bootstrap_pr_auc.png")
        P.calibration(yte, test_p["XGBoost"], p_test, fd / "06_calibration.png")
        P.cost_curve(val_sweep, thresholds["cost_optimal_global"],
                     {"90% recall": thresholds["recall90_global"]}, fd / "07_cost_curve.png")
        P.policy_savings(test_rows, fd / "08_policy_savings.png")
        P.confusion_pair({k: test_rows[k] for k in ("Default 0.5", "Cost-optimal threshold", "Expected-cost + guardrail")}, fd / "09_confusion_matrices.png")
        P.shap_bar(sh["shap_values"], sh["sample"], fd / "10_shap_importance.png")
        P.shap_beeswarm(sh["shap_values"], sh["sample"], fd / "11_shap_beeswarm.png")
        P.shap_waterfall(sh["caught"][0], "Highest-scored fraud in the test set", fd / "12_shap_waterfall_caught.png")
        if sh["missed"]:
            P.shap_waterfall(sh["missed"][0], "A fraud the model missed", fd / "13_shap_waterfall_missed.png")
        plt.close("all")

        rd = C.RESULTS_DIR
        (rd / "metrics.json").write_text(json.dumps(results, indent=2))
        comparison.to_csv(rd / "model_comparison.csv", index=False)
        grid.to_csv(rd / "xgb_grid_search.csv", index=False)
        sens.to_csv(rd / "cost_sensitivity.csv", index=False)
        test_sweep.to_csv(rd / "test_threshold_sweep.csv", index=False)
        pd.DataFrame(test_rows).T.to_csv(rd / "test_policies.csv")
        C.MODEL_DIR.mkdir(exist_ok=True)
        xgb_model.save_model(C.MODEL_DIR / "xgb_model.json")

    return results
