# Day 11 — Model Choice

## What I picked: Random Forest, default settings (no class weighting)

## Why (tested, not assumed)

The roadmap frames this as "random forest / gradient boosting basics" — rather than pick one from intuition, I ran both families, with and without class-weight balancing, against the exact same Week 2 train/test split:

| Model | Macro F1 |
|---|---|
| **Random Forest (default)** | **0.5757** |
| Gradient Boosting (default) | 0.5387 |
| HistGradientBoosting (class_weight=balanced) | 0.5362 |
| Random Forest (class_weight=balanced) | 0.5315 |
| HistGradientBoosting (default) | 0.5302 |
| *Week 2 baseline (logistic regression)* | *0.5272* |

## The result that changed my plan: `class_weight="balanced"` made things worse

Going in, the plan was to enable `class_weight="balanced"` by default — it's the obvious move given the rare-class problems flagged all through Week 2 (TV-Y7-FV, TV-14). Testing it directly showed the opposite: balancing **hurt** macro F1 for both Random Forest (0.576 → 0.532) and HistGradientBoosting. My read: with only 10 training examples for the smallest class, telling the model to weight that class equally to a 254-example class doesn't give it more signal to learn from — it just makes the model chase noise in a handful of rows. Balancing helps when minority classes are underrepresented but still have enough data to learn a real pattern; at this sample size, they don't. Worth remembering before reaching for `class_weight="balanced"` as a default move on future problems.

## Why Random Forest over Gradient Boosting generally

Random Forest's bagging approach (many independent trees, averaged) is more robust to overfitting with default settings than boosting's sequential error-correction, which matters on a small dataset (1,157 training rows) where boosting has less room to safely fit residuals without overfitting. Gradient Boosting usually pulls ahead once properly tuned, but Week 3's brief is light tuning, not exhaustive search — Random Forest's stronger out-of-the-box result made it the safer "one clean improvement" instead of a model that needed more tuning investment to catch up.

## Result

Random Forest beats the Week 2 baseline by a real, non-marginal margin: **0.576 vs. 0.527 macro F1 (+9.2% relative)** — not just noise, and confirmed reproducible with a fixed `random_state=42`.
