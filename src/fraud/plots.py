"""All figures. Each function saves a PNG and returns the matplotlib Figure."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

COLORS = {"Logistic regression": "#6c757d", "Random forest": "#2a9d8f", "XGBoost": "#e76f51"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return fig


def class_balance(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["Class"].value_counts().sort_index()
    ax[0].bar(["Legit", "Fraud"], counts.values, color=["#457b9d", "#e63946"])
    ax[0].set_title("Class counts (linear scale)")
    ax[1].bar(["Legit", "Fraud"], counts.values, color=["#457b9d", "#e63946"])
    ax[1].set_yscale("log"); ax[1].set_title("Class counts (log scale)")
    for a in ax:
        for i, v in enumerate(counts.values):
            a.text(i, v, f"{v:,}", ha="center", va="bottom")
    return _save(fig, path)


def amount_and_time(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for cls, name, c in [(0, "Legit", "#457b9d"), (1, "Fraud", "#e63946")]:
        sub = df[df["Class"] == cls]
        ax[0].hist(np.log1p(sub["Amount"]), bins=50, density=True, alpha=0.55, label=name, color=c)
        ax[1].hist(sub["Time"] / 3600, bins=48, density=True, alpha=0.55, label=name, color=c)
    ax[0].set(title="log(1 + Amount) by class", xlabel="log(1 + Amount EUR)", ylabel="density")
    ax[1].set(title="Transaction time by class", xlabel="hours since first transaction", ylabel="density")
    ax[0].legend(); ax[1].legend()
    return _save(fig, path)


def split_timeline(df_sorted, cut_hours, path):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    hours = df_sorted["Time"] / 3600
    ax.hist(hours, bins=96, color="#adb5bd", label="all transactions")
    ax2 = ax.twinx()
    ax2.hist(hours[df_sorted["Class"] == 1], bins=96, color="#e63946", alpha=0.7, label="frauds")
    for h, lab in zip(cut_hours, ["train | val", "val | test"]):
        ax.axvline(h, color="black", ls="--"); ax.text(h, ax.get_ylim()[1] * 0.92, f" {lab}", fontsize=9)
    ax.set(xlabel="hours since first transaction", ylabel="transactions", title="Chronological split (60 / 20 / 20)")
    ax2.set_ylabel("frauds", color="#e63946")
    return _save(fig, path)


def pr_roc(y, probs: dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, p in probs.items():
        pr, rc, _ = precision_recall_curve(y, p)
        from sklearn.metrics import average_precision_score, roc_auc_score
        ax[0].step(rc, pr, where="post", color=COLORS.get(name), label=f"{name} (PR-AUC {average_precision_score(y, p):.3f})")
        fpr, tpr, _ = roc_curve(y, p)
        ax[1].plot(fpr, tpr, color=COLORS.get(name), label=f"{name} (ROC-AUC {roc_auc_score(y, p):.3f})")
    ax[0].axhline(np.mean(y), color="grey", ls=":", label=f"random ({np.mean(y):.4f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test set)")
    ax[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (test set)")
    ax[0].legend(loc="lower left"); ax[1].legend(loc="lower right")
    return _save(fig, path)


def bootstrap_hist(boot: dict, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, b in boot.items():
        ax.hist(b["samples"], bins=40, alpha=0.5, color=COLORS.get(name),
                label=f"{name}: 95% CI [{b['ci_low']:.3f}, {b['ci_high']:.3f}]")
    ax.set(xlabel="test PR-AUC", ylabel="bootstrap resamples", title="Bootstrap distribution of test PR-AUC")
    ax.legend()
    return _save(fig, path)


def calibration(y, p_raw, p_cal, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    for p, lab, c in [(p_raw, "raw XGBoost", "#e76f51"), (p_cal, "Platt-calibrated", "#264653")]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, "o-", label=lab, color=c)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    ax.set(xscale="log", yscale="log", xlabel="mean predicted probability", ylabel="observed fraud rate",
           title="Reliability curve (test set, log-log)")
    ax.legend()
    return _save(fig, path)


def cost_curve(sweep_df, chosen, others: dict, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    s = sweep_df.sort_values("threshold")
    ax[0].plot(s["threshold"], s["total_cost_eur"], color="#264653")
    ax[0].axhline(s["fraud_value_eur"].iloc[0], color="grey", ls=":", label="no model (all fraud lost)")
    ax[0].axvline(chosen, color="#e63946", ls="--", label=f"cost-optimal ({chosen:.3f})")
    for lab, t in others.items():
        ax[0].axvline(t, ls=":", color="black", alpha=0.6); ax[0].text(t, ax[0].get_ylim()[1] * 0.9, f" {lab}", fontsize=8)
    ax[0].set(xscale="log", yscale="log", xlabel="threshold (calibrated probability, log)", ylabel="total cost (EUR, log)",
              title="Validation cost vs threshold")
    ax[0].legend()
    ax[1].plot(s["threshold"], s["recall"], label="recall", color="#e76f51")
    ax[1].plot(s["threshold"], s["precision"], label="precision", color="#2a9d8f")
    ax[1].axvline(chosen, color="#e63946", ls="--")
    ax[1].set(xscale="log", xlabel="threshold (log)", ylabel="score", title="Validation precision / recall vs threshold")
    ax[1].legend()
    return _save(fig, path)


def confusion_pair(results: dict, path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.2))
    for ax, (lab, r) in zip(axes, results.items()):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["pred legit", "pred fraud"], yticklabels=["true legit", "true fraud"])
        ax.set_title(f"{lab}\nrecall {r['recall']:.1%} | precision {r['precision']:.1%}", fontsize=10)
    return _save(fig, path)


def shap_bar(shap_values, X, path):
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=15)
    fig = plt.gcf(); return _save(fig, path)


def shap_beeswarm(shap_values, X, path):
    plt.figure()
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    fig = plt.gcf(); return _save(fig, path)


def shap_waterfall(explanation, title, path):
    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=12)
    fig = plt.gcf(); fig.suptitle(title, fontsize=11)
    return _save(fig, path)


def policy_savings(test_rows: dict, path):
    names = list(test_rows)
    sav = [test_rows[n]["saving_vs_no_model_pct"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#e63946" if v < 0 else "#2a9d8f" for v in sav]
    bars = ax.barh(names, sav, color=colors)
    for b, n, v in zip(bars, names, sav):
        r = test_rows[n]
        ax.text(max(v, 0) + 3, b.get_y() + b.get_height() / 2,
                f"{v:+.1f}%  ({r['alerts']:,} alerts, recall {r['recall']:.0%})",
                va="center", ha="left", fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlim(min(min(sav) * 1.1, 0), 200)
    ax.invert_yaxis()
    ax.set(xlabel="loss avoided vs. no model (%, test set, review cost EUR 5)", title="Decision policies on the test set")
    return _save(fig, path)
