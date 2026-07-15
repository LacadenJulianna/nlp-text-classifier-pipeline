# nlp-text-classifier-pipeline

A lightweight, end-to-end NLP pipeline for Spam/Harm detection and IMDB sentiment
analysis. Built with Python, scikit-learn, and pandas using classical ML models
(TF-IDF + Logistic Regression/Naive Bayes) and deployed as an interactive live
dashboard via Streamlit.

This repo contains two projects, built in sequence as part of an SLU internship:

## `spam-harm/` — Spam/Harm text classifier

The actual deliverable described above. TF-IDF + classical ML (Naive Bayes
baseline, tuned Logistic Regression candidate), full pipeline from raw data to
a deployable Streamlit app.

See [`spam-harm/README.md`](spam-harm/README.md) for setup, usage, model
accuracy, and known limitations.

## `stage01/` – `stage04/` — Disney+ content-rating classifier

A fundamentals practice project completed before `spam-harm/`: predicting a
Disney+ title's content rating (G, PG, TV-14, etc.) from catalog metadata
(genre, runtime, release year) using a tuned Random Forest, deployed live via
Streamlit.

See [`stage04/README.md`](stage04/README.md) for setup, usage, model
accuracy, and known limitations.

## Project structure

```
spam-harm/    Spam/Harm text classifier (the repo's primary deliverable)
  data/       Raw dataset, cleaning pipeline, canonical train/test split
  eda/        Class balance, message length, and token-frequency exploration
  baseline/   TF-IDF + Naive Bayes baseline model and metrics
  iterate/    Tuned Logistic Regression candidate, comparison, final model
  app/        predict.py, Streamlit app, stress tests, deployment guide

stage01/      Disney+ project: data exploration and cleaning
stage02/      Disney+ project: feature engineering and baseline model
stage03/      Disney+ project: model iteration (Random Forest, tuning)
stage04/      Disney+ project: packaging, interface, deployment
```
