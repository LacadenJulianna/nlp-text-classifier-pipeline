# Disney+ Content Rating Predictor

Predicts a likely content rating (TV-G, PG, TV-14, etc.) for a Disney+
title from its genre, runtime, and release year. Built as a 4-week
learning project: data exploration → baseline model → iteration →
deployment.

**Live demo:** [add your Streamlit Cloud URL here once deployed — see DEPLOYMENT.md]

## Setup (running locally)

```bash
cd stage04
pip install -r requirements.txt
streamlit run app.py
```

Requires `stage03/final_model.joblib` and `stage02/features/feature_pipeline.joblib`
to exist — both are produced by earlier stages of this pipeline (see their
respective READMEs/workflow docs) and are committed to this repo directly
so the app works without re-running the full training pipeline.

## Usage

1. Pick **Movie** or **TV Show**.
2. Enter runtime in minutes (movies) or number of seasons (TV shows).
3. Enter the release year.
4. Select any genres that apply — anything not in the list gets treated
   as "Other," the same way it was during training.
5. Click **Predict rating**.

The result shows the predicted rating, a confidence score, and the full
probability breakdown across all 9 possible ratings.

## What this model can do

- Trained on 1,447 real Disney+ catalog titles, evaluated on a held-out
  test set it never saw during training.
- Final model: Random Forest, macro F1 = 0.596 on the test set — a real,
  verified 13.1% improvement over a simple logistic regression baseline
  (0.527) and a 15x improvement over guessing the most common rating
  every time (0.040).
- Handles genre combinations reasonably well — genre was the strongest
  single signal found during exploration (Week 1, Day 4).

## What this model can't do — read this before trusting a prediction

- **Weakest on TV-14 specifically.** Recall for TV-14 is 0.06 in the
  final model — it correctly identifies only about 1 in 16 real TV-14
  titles, most often confusing it with TV-PG. This isn't a minor edge
  case; TV-14 is a meaningful content boundary. If you're testing this
  model, this is the rating to be most skeptical of.
- **Struggles with rare classes generally.** TV-Y7-FV had only 13 total
  examples in the entire dataset (3 in the test set) — there simply
  isn't enough data for any model to learn this class reliably.
- **Doesn't use `country`.** Tested and discarded during Week 3 (gain
  was within noise for the sample size available) — a title's country
  of origin has no effect on this model's prediction.
- **Doesn't understand rating order.** TV-Y < TV-Y7 < TV-G < TV-PG <
  TV-14 is a real progression, but this model treats all 9 ratings as
  unrelated categories — a prediction that's "one step off" (e.g.
  TV-PG instead of TV-14) is scored exactly the same as a wildly wrong
  one. This is a known, documented limitation (Week 2, Day 6), not an
  oversight.
- **Doesn't know about `director` or `cast`.** Excluded early (Week 1)
  due to high missing data and high cardinality — a title's specific
  cast/director has no bearing on this model's prediction, even though
  it plausibly could in reality.
- **Trained on a specific catalog snapshot.** Predictions reflect
  patterns in this dataset as of when it was collected — new titles,
  new genres, or changes in how Disney+ assigns ratings over time
  aren't reflected here.

## Project structure

```
stage01/  Data exploration and cleaning
stage02/  Feature engineering and baseline model
stage03/  Model iteration (Random Forest, tuning)
stage04/  This app: packaging, interface, deployment
```

See each stage's own documentation (`workflow.md`, `findings.md`,
`problem_statement.md`, `progress_report.md`) for the full reasoning
behind each decision referenced above.

## Known open items (not yet resolved)

- Dataset license confirmation (flagged since Stage 01, still open).
- Whether the TV-14 regression from Week 3's tuning is acceptable for
  any real use beyond this learning project — not decided here.
