# Spam/Harm Text Classifier

Classifies a text message as spam or ham (legitimate) using TF-IDF +
classical ML (Naive Bayes baseline, tuned Logistic Regression as the
final model — whichever won on held-out spam F1, see
`iterate/model_choice.md`). Trained on a small labeled spam/ham dataset
(1,000 raw rows; 677 after removing exact-duplicate messages).

**Live demo:** [add your Streamlit Cloud URL here once deployed — see app/DEPLOYMENT.md]

## Setup (running locally)

```bash
cd spam-harm/app
pip install -r requirements.txt
streamlit run app.py
```

Requires `spam-harm/iterate/final_model.joblib` to exist (committed to
this repo directly, produced by `iterate/evaluate_candidate.py`).

## Usage

1. Paste any message into the text box.
2. Click **Classify**.
3. See the predicted label (spam/ham), a confidence score, and the full
   probability breakdown.

## What this model can do

- Trained on 677 labeled messages (after deduplication), evaluated on a held-out test set.
- Headline metric is **spam-class F1** (see `iterate/model_choice.md`
  for the exact number) — not accuracy, since the deduplicated dataset is
  ~74% ham and accuracy alone would reward a model that just says "ham" every time.
- Beats the naive baseline (always predict ham, spam F1 = 0.0) by a
  real, verified margin (see `baseline/metrics_report.md`).

## What this model can't do — read this before trusting a prediction

- Trained on a small (677-message, post-dedup), possibly synthetic
  dataset of longer email-style text, not real-world SMS traffic —
  performance on genuinely different text (short SMS, social media
  DMs, etc.) is untested.
- English only — the training data is English-language text.
- "Harm" here means spam/unsolicited-commercial text specifically, not
  a general toxicity/harassment classifier — those are a different
  labeling task with different training data.
- Doesn't use stemming (see `iterate/tuning_log.md` for why this was
  discarded for this project's scope) — messages with heavy misspelling
  or unusual word forms may be harder to classify correctly.

## Project structure

```
data/       Raw dataset, cleaning pipeline, canonical train/test split
eda/        Class balance, message length, and token-frequency exploration
baseline/   TF-IDF + Naive Bayes baseline model and metrics
iterate/    Tuned Logistic Regression candidate, comparison, final model
app/        predict.py, Streamlit app, stress tests, deployment guide
```

## Known open items (not yet resolved)

- Dataset provenance/license unconfirmed — the source of `spam-harm/data/raw/spam.csv` (downloaded from Kaggle to `C:\Users\Julianna\Downloads\spam_dataset.csv`) and whether it's a real or synthetic dataset was not verified. Confirm before any use beyond this learning project.
- Stemming was considered and discarded for this iteration (see `iterate/tuning_log.md`) — worth revisiting if a future iteration shows token sparsity is a real bottleneck.
