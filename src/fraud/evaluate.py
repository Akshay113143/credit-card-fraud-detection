"""Ranking metrics and bootstrap confidence intervals."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ranking_metrics(y, p) -> dict:
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def precision_at_k(y, p, k: int) -> float:
    """Share of real frauds among the k highest-scored transactions (an analyst's daily queue)."""
    y = np.asarray(y); top = np.argsort(-np.asarray(p), kind="mergesort")[:k]
    return float(y[top].mean())


def bootstrap_pr_auc(y, p_by_model: dict, n: int, seed: int) -> dict:
    """Resample test rows with replacement; report 95% percentile intervals.

    The same resampled rows are used for every model, so the difference
    between two models is a paired comparison.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y); n_rows = len(y)
    scores = {m: [] for m in p_by_model}
    done = 0
    while done < n:
        idx = rng.integers(0, n_rows, n_rows)
        if y[idx].sum() == 0:
            continue
        for m, p in p_by_model.items():
            scores[m].append(average_precision_score(y[idx], np.asarray(p)[idx]))
        done += 1
    out = {}
    for m, s in scores.items():
        s = np.array(s)
        out[m] = {"mean": float(s.mean()), "ci_low": float(np.percentile(s, 2.5)),
                  "ci_high": float(np.percentile(s, 97.5)), "samples": s}
    return out
