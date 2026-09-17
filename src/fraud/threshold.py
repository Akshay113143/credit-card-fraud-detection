"""Turning scores into decisions.

Three kinds of policy are compared:
  1. a single global threshold (default 0.5, a recall target, or cost-optimal on validation)
  2. the expected-cost rule: flag when P(fraud) x Amount >= review cost
  3. the expected-cost rule plus a high-confidence guardrail (flag anyway when P(fraud) >= 0.5)
"""
import numpy as np
import pandas as pd


def evaluate_flags(y, flags, amount=None, review_cost=None) -> dict:
    """Confusion counts, precision/recall and (optionally) euro cost for a boolean flag vector."""
    y = np.asarray(y).astype(int); f = np.asarray(flags).astype(bool)
    tp = int((f & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
    fn = int((~f & (y == 1)).sum()); tn = int((~f & (y == 0)).sum())
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "alerts": tp + fp,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
    }
    if amount is not None:
        a = np.asarray(amount, dtype=float)
        fraud_total = a[y == 1].sum()
        caught = a[f & (y == 1)].sum()
        review = review_cost * f.sum()
        missed = fraud_total - caught
        out.update({
            "fraud_value_eur": float(fraud_total),
            "value_recall": float(caught / fraud_total) if fraud_total else 0.0,
            "review_cost_eur": float(review),
            "missed_fraud_eur": float(missed),
            "total_cost_eur": float(review + missed),
            "saving_vs_no_model_pct": float(100 * (1 - (review + missed) / fraud_total)) if fraud_total else 0.0,
        })
    return out


def confusion_at(y, p, t):
    r = evaluate_flags(y, np.asarray(p) >= t)
    return r["tp"], r["fp"], r["fn"], r["tn"]


def metrics_at(y, p, t, amount=None, review_cost=None) -> dict:
    return {"threshold": float(t), **evaluate_flags(y, np.asarray(p) >= t, amount, review_cost)}


def expected_cost_flags(p, amount, review_cost):
    """Flag when the expected loss avoided (P x Amount) is at least the cost of a review.

    This is the Bayes-optimal decision for this cost model, but only if p is a real
    probability - which is why the model is calibrated first.
    """
    return np.asarray(p) * np.asarray(amount, dtype=float) >= review_cost


def guarded_flags(p, amount, review_cost, confidence=0.5):
    """Expected-cost rule, plus: always flag very likely fraud even if the amount is tiny.

    Small 'card-testing' charges often precede larger fraud, so ignoring them purely on
    amount is risky. 0.5 is a fixed business rule, not a tuned value.
    """
    return expected_cost_flags(p, amount, review_cost) | (np.asarray(p) >= confidence)


def candidate_thresholds(p) -> np.ndarray:
    return np.unique(np.concatenate([np.unique(p), [1.01]]))


def cost_optimal_threshold(y, p, amount, review_cost) -> float:
    """Global threshold minimising review_cost x alerts + missed fraud amount (vectorised)."""
    y = np.asarray(y); p = np.asarray(p, dtype=float); amount = np.asarray(amount, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    p_s, y_s, a_s = p[order], y[order], amount[order]
    fraud_amt_cum = np.concatenate([[0.0], np.cumsum(a_s * (y_s == 1))])
    k = np.arange(len(p_s) + 1)
    cost = review_cost * k + (fraud_amt_cum[-1] - fraud_amt_cum)
    valid = np.ones_like(cost, dtype=bool)
    valid[1:-1] = p_s[:-1] != p_s[1:]          # can only cut between distinct scores
    best_k = int(k[valid][np.argmin(cost[valid])])
    return 1.01 if best_k == 0 else float(p_s[best_k - 1])


def threshold_for_recall(y, p, target: float) -> float:
    """Highest threshold whose recall is still >= target."""
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    fraud_scores = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(fraud_scores)))
    return float(fraud_scores[k - 1])


def sweep(y, p, thresholds, amount=None, review_cost=None) -> pd.DataFrame:
    return pd.DataFrame([metrics_at(y, p, t, amount, review_cost) for t in thresholds])
