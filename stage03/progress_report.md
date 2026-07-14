# Day 15 — Progress Report: Baseline vs. Final Model

## Comparison table

| Model | Macro F1 | Accuracy | vs. Week 2 baseline |
|---|---|---|---|
| Week 2 baseline (logistic regression) | 0.5272 | 61.7% | — |
| Day 12 candidate (Random Forest, untuned) | 0.5757 | 60.0% | +9.2% |
| **Day 14 final (Random Forest, tuned)** | **0.5962** | **63.1%** | **+13.1%** |

Tuned params: `max_depth=None, min_samples_leaf=2, n_estimators=200` (found via 3-fold CV grid search on training data only — test set was touched exactly once, at the end, to report this number).

## The honest complication: tuning improved the average, but made the worst class worse

TV-14 — already the weakest class at baseline (0.26 F1) — dropped further under the tuned final model:

| Model | TV-14 F1 | TV-14 recall | Where TV-14 titles actually got predicted |
|---|---|---|---|
| Baseline | 0.26 | 0.19 | 9/16 → TV-PG |
| Day 12 candidate | 0.28 | 0.25 | 10/16 → TV-PG |
| **Day 14 final (tuned)** | **0.11** | **0.06** | **11/16 → TV-PG** |

The overall macro F1 went up because other classes (G, TV-Y, TV-Y7) improved enough to outweigh this regression in the average — that's exactly the risk of optimizing a single averaged number without checking what happened underneath it. The tuned model is a better model *overall*, but it is a **worse** model specifically for the class most likely to matter if this were ever used for real content moderation (TV-14 is a meaningful boundary rating, not a minor edge case like TV-Y7-FV). Worth saying this plainly in the demo rather than only showing the headline 0.596.

## Why this happened (plausible explanation, not confirmed)

TV-14's confusion is concentrated on its immediate ordinal neighbor, TV-PG — the same pattern flagged in Week 2's Day 9 evaluation. Tuning toward `min_samples_leaf=2` likely makes the trees slightly more conservative/generalized, which may disproportionately hurt a class that was already borderline between two adjacent categories. This is a hypothesis, not something confirmed by further testing this week — flagging it as a real open question rather than a settled explanation.

## What was tried and discarded (Day 13)

Two feature additions were tested against Day 12's model: adding `country` (+0.006 macro F1) and `genre_count` (+0.003 macro F1). Both gains were below a 0.02 meaningful-gain threshold on a 290-row test set — likely noise, not real signal — so both were discarded per the roadmap's own instruction not to chase small gains. Combining them actually hurt performance substantially (-0.079), also discarded.

## Is it good enough to deploy?

Per the roadmap's own bar ("doesn't need to be perfect, needs to be done"): yes, with the TV-14 caveat stated openly rather than hidden. The model beats the Week 2 baseline by a real margin, the process to get here is fully reproducible (fixed random seed, verified split reproduction, tuning done without touching the test set until the final check), and the one known weakness is documented with an actual number and a plausible cause — not glossed over.

## Open items carried forward (not resolved this week)

- TV-14's regression under tuning — worth a dedicated look before this goes anywhere near production use.
- `country` and `genre_count`: discarded for this model, but worth re-testing if the feature set changes substantially later (a marginal feature can become useful in combination with a different model or after a different fix).
- Dataset license (open since Stage 01) — still unresolved.
