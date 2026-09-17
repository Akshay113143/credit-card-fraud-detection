# Interview notes: credit card fraud detection

These are direct answers to the questions this project is likely to prompt. Every number comes from
`results/metrics.json`.

## Key numbers to remember

| | |
|---|---|
| Raw data | 284,807 transactions, 492 frauds (0.173%), 48 hours |
| After de-duplication | 283,726 rows, 473 frauds (1,081 duplicates removed, 19 of them fraud) |
| Split | chronological 60/20/20 → 342 / 57 / 74 frauds |
| scale_pos_weight | 496.8 |
| Best XGBoost | max_depth 5, min_child_weight 5, 400 trees, lr 0.05 |
| Test PR-AUC | XGBoost 0.798 (CI 0.715–0.877) · RF 0.809 · LogReg 0.745 |
| Test ROC-AUC | XGBoost 0.970 · RF 0.965 · LogReg 0.981 |
| Default 0.5 | 56/74 caught, 18 false positives, 61.1% saving |
| Expected-cost rule | 78 alerts, 25/74 caught, 76.6% of fraud € caught, **71.6% saving** |
| + guardrail | 115 alerts, 57/74 caught (77.0%), 69.7% saving |
| 90% recall target | 4,958 alerts, −243% saving |
| Brier score | 0.00054 → 0.00040 after Platt scaling |
| Top SHAP features | V4 1.85, V14 1.74, V12 1.22, V11 0.80, V10 0.74 |

## Terms, in one line each

- **Precision** = TP / (TP + FP): of the transactions flagged, how many were fraud.
- **Recall** = TP / (TP + FN): of all frauds, how many were flagged.
- **PR-AUC / average precision** = area under the precision-recall curve, computed as
  Σ (Rₙ − Rₙ₋₁) · Pₙ over thresholds. Random baseline = positive rate.
- **ROC-AUC** = probability that a random fraud scores higher than a random legit transaction. Random baseline = 0.5.
- **Brier score** = mean of (p − y)². Measures calibration and sharpness together; lower is better.
- **Calibration**: a model is calibrated if, among transactions scored 0.1, about 10% are fraud.
- **Platt scaling**: fit σ(a·logit(p) + b) on held-out data to recalibrate scores.
- **Bootstrap CI**: resample the test set with replacement many times, recompute the metric, and take the
  2.5th and 97.5th percentiles.
- **SHAP value**: a feature's Shapley contribution to one prediction. Contributions add up exactly to
  (model output − average output).
- **TreeExplainer**: an exact, polynomial-time SHAP algorithm for tree ensembles.
- **Log-odds**: log(p / (1 − p)). This is XGBoost's raw output for binary classification.

## Likely questions and answers

**1. Why not accuracy?**
A model that says "legit" for everything is 99.83% accurate and catches zero fraud. Accuracy is dominated
by the majority class.

**2. Why PR-AUC and not ROC-AUC?**
ROC uses FPR = FP / (FP + TN). With about 56,700 legitimate test transactions, even 2,872 false alarms give
an FPR of only 5%. In this project logistic regression had the *highest* ROC-AUC (0.981) but the *lowest*
PR-AUC (0.745). Precision has no TN term, so it shows the flood of false alarms directly.

**3. What does scale_pos_weight do mathematically?**
XGBoost builds each tree from the gradient gᵢ and Hessian hᵢ of the log-loss. With scale_pos_weight = w,
every positive row's gᵢ and hᵢ are multiplied by w. Split gains and leaf weights
(−Σg / (Σh + λ)) are then computed as if each fraud appeared w times. I set w = negatives / positives
in the training block = 496.8.

**4. Why not SMOTE?**
SMOTE creates synthetic frauds by interpolating between a fraud and its nearest fraud neighbours. With 342
training frauds in 31 dimensions, those neighbours can belong to different fraud patterns, so the synthetic
points may land where no real fraud exists. Reweighting changes only the loss, not the data. SMOTE must
also be applied only inside the training fold, otherwise it leaks test information.

**5. Why a chronological split?**
A deployed model scores future transactions. A random split lets the model learn from transactions that
happened after the ones it is tested on. Here the random split was only 1.9 PR-AUC points more optimistic
(0.817 vs 0.798), which is inside the confidence interval, but the time split is the honest set-up.

**6. Why did you drop Time?**
Raw Time is seconds since the first transaction. In a 48-hour dataset it mostly encodes "day 1 or day 2",
which does not carry over to future data. I kept the time of day as sin(2π·h/24) and cos(2π·h/24) so that
hour 23.9 and hour 0.1 end up close together.

**7. Why remove duplicates?**
1,081 rows are identical in every column. Identical rows in train and test let the model score
memorised examples, which inflates results.

**8. Random forest scored higher on test. Why did you keep XGBoost?**
Model selection happened on validation, where XGBoost led (0.787 vs 0.765). On test, RF is 0.011 higher,
but the paired bootstrap difference has a 95% CI of −0.040 to +0.014, so the two are tied. Switching
after seeing test results turns the test set into a second validation set.

**9. How did you compute the confidence intervals?**
1,000 bootstrap resamples of the 56,746 test rows. The same resampled indices are used for every model,
so model differences are paired. Resamples with no fraud are skipped. The interval is the 2.5th to
97.5th percentile.

**10. Why calibrate if PR-AUC does not change?**
The decision rule multiplies probability by amount, so it needs real probabilities. scale_pos_weight shifts
raw scores upward: a raw 0.5 is really about an 8% fraud probability. Platt scaling fixed this: the Brier
score fell from 0.00054 to 0.00040, and the mean predicted probability (0.0012) now matches the test fraud
rate (0.0013). Platt scaling is monotonic, so rankings and PR-AUC stay the same.

**11. Explain the expected-cost rule.**
Reviewing a transaction costs c (€5). Not reviewing a fraud loses its amount A. The expected loss avoided
by reviewing is P(fraud) · A, so review exactly when P · A ≥ c. This is the Bayes-optimal decision for
this cost model. It needs no tuned threshold, only calibrated P.

**12. The expected-cost rule has 34% recall. Isn't that bad?**
It catches 25 of 74 fraud cases but 76.6% of fraud euros, and saves the most money (71.6%). The frauds it
skips are mostly tiny: 42 of the 49 are under €10, with a median of €1. Because tiny charges can be card testing that comes before bigger fraud, I added a
guardrail (also flag if P ≥ 0.5). That raises case recall to 77.0% at a cost of 1.9 points of saving and 37
more alerts.

**13. Why did the 90% recall target lose money?**
Reaching 90% recall on validation needed a threshold of 0.00046. On test that flagged 4,958 transactions,
which is €24,790 of review cost, to protect €7,728 of fraud. Recall targets ignore what a false alarm costs.

**14. Why did the "cost-optimal threshold" do worse on test than the default 0.5?**
It was fitted on 57 validation frauds. The validation cost curve is almost flat from 0.01 to 0.7, so the exact
minimum (0.674) is mostly noise. On test it caught 4 fewer frauds than the default and saved 48.9% vs 61.1%.
Lesson: optimising one number on a few positives overfits. A rule based on probability theory is more stable.

**15. How sensitive is this to the €5 assumption?**
I re-ran every policy at €1, 2, 5, 10 and 20. The expected-cost rule beat the single threshold at every
level (for example 57.2% vs 38.2% at €20). Its alert count changes on its own (327 alerts at €1, 28 at €20).

**16. How do you read a SHAP waterfall?**
Start at E[f(x)] (2.48 log-odds, which is high because of scale_pos_weight). Each bar adds a feature's
contribution, and the total ends at f(x). For the highest-scored test fraud, V14 = −4.29 added +3.97
log-odds, which multiplies the fraud odds by e^3.97 ≈ 53.

**17. What did SHAP show about the missed fraud?**
It scored P = 0.00008. V11, V14 and V8 all pushed toward "legit" (−2.26, −2.19, −1.78). Nothing in these
features separates it from normal traffic, so no threshold catches it cheaply. You would need new
features, such as per-card velocity.

**18. Can you interpret V14?**
No. It is a PCA component of undisclosed original features. SHAP can tell you that low V14 values drive
fraud predictions, which is useful for auditing and monitoring drift, but not what V14 means in business
terms.

**19. What would you do next with real data?**
Build velocity and aggregate features per card and merchant. Use a longer time window with rolling
retraining. Put analyst capacity into the decision as an alert budget. Add chargeback fees and
false-decline churn to the cost model. Use separate blocks for tuning, calibration and threshold setting.
Monitor score drift in production.

**20. Isn't this dataset over-used?**
Yes, so the value is in the method rather than the model: a time-based split, de-duplication, confidence
intervals, calibration, and choosing decisions by euros saved instead of by recall.

## Code points to be able to explain

```python
# features.py — cyclical hour
hour = (df["Time"] % 86400) / 3600.0          # seconds -> hour of day (0–24)
X["hour_sin"] = np.sin(2 * np.pi * hour / 24)  # position on a circle
X["hour_cos"] = np.cos(2 * np.pi * hour / 24)  # second coordinate on the circle
```
Using sin alone, hours 3 and 9 get the same value. Adding cos gives every hour a unique point on the circle.

```python
# threshold.py — cost-optimal threshold in O(n log n)
order = np.argsort(-p)                                   # highest score first
fraud_amt_cum = np.concatenate([[0.0], np.cumsum(a_s * (y_s == 1))])
cost = review_cost * k + (fraud_amt_cum[-1] - fraud_amt_cum)   # flag the top k rows
valid[1:-1] = p_s[:-1] != p_s[1:]                        # tied scores must be flagged together
```
Flagging the top k rows costs k·c for reviews plus the fraud amount outside the top k. A cumulative sum
evaluates every k in one pass instead of looping over thresholds. A unit test checks the result against
brute force.

```python
# models.py — Platt scaling
logit = np.log(p / (1 - p))                    # back to log-odds
LogisticRegression(C=1e6).fit(logit, y_val)    # learns a, b in σ(a·logit + b); C large = no regularisation
```

```python
# evaluate.py — paired bootstrap
idx = rng.integers(0, n_rows, n_rows)          # one resample, shared by all models
average_precision_score(y[idx], p[idx])
```

```python
# threshold.py — decision rules
p * amount >= review_cost                          # expected-cost rule
(p * amount >= review_cost) | (p >= 0.5)           # + guardrail
```
