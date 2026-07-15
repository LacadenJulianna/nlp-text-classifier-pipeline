# Spam/Harm Text Classifier — Design

## Context

The repo's README describes the intended deliverable: "a lightweight, end-to-end NLP pipeline for Spam/Harm detection and IMDB sentiment analysis... classical ML models (TF-IDF + Logistic Regression/Naive Bayes) deployed as an interactive live dashboard via Streamlit." `stage01`-`stage04` were the fundamentals practice run (a Disney+ content-rating classifier using structured/tabular metadata, not text) that preceded this real deliverable.

Spam/Harm detection and IMDB sentiment analysis are separate subsystems (different datasets, different label spaces). This spec covers **Spam/Harm detection only**, built and shipped first. IMDB sentiment analysis is a separate future spec.

## Goal

Build a spam/harm SMS text classifier — TF-IDF + classical ML — through the full arc from raw data to a live Streamlit demo, with the same rigor and conventions established in `stage01`-`stage04` (honest metric reporting, documented tradeoffs, stress-testing, verified deployment).

## Folder structure

New top-level folder `spam-harm/`, sibling to `stage01`-`stage04` (not a continuation of that day/stage numbering, since there's no external day-by-day roadmap for this deliverable):

```
spam-harm/
  data/
    raw/          raw SMS Spam Collection dataset (user-supplied, not fetched by the assistant)
    clean_data.py cleaning script -> cleaned dataset
  eda/
    eda.py or eda.ipynb
    findings.md
  baseline/
    train_baseline.py
    baseline_model.joblib
    metrics_report.md
    confusion_matrix.png
  iterate/
    train_candidate.py
    tune_final.py
    tuning_log.md
    model_choice.md
    final_model.joblib
  app/
    predict.py
    app.py
    stress_test.py
    requirements.txt
    DEPLOYMENT.md
  README.md
```

## Data component

**Dataset:** SMS Spam Collection (~5,574 SMS messages, `spam`/`ham` binary labels), a well-known small well-labeled public dataset. The user sources the raw file themselves and places it in `spam-harm/data/raw/`; the assistant does not fetch external files directly. Loading code expects a simple two-column format (`label`, `message`) and adapts if the actual download's format differs.

**Cleaning (`spam-harm/data/clean_data.py`):**
- Load raw file, standardize column names, map `spam`/`ham` to a binary label.
- Basic text cleaning: lowercase, strip punctuation/whitespace noise. Do not apply aggressive stemming/stopword removal by default — test whether it helps as an EDA/iteration question rather than assuming it does.
- Check duplicates, nulls, and class balance. This dataset is known to be imbalanced (~87% ham / ~13% spam) — confirm and document, since it drives metric choice.

**EDA (`spam-harm/eda/`):**
- Class balance chart, message-length distribution by class, most distinctive tokens per class.
- `findings.md` documenting what was learned, same convention as `stage01/findings.md`.

## Model component

**Baseline (`spam-harm/baseline/`):**
- Pipeline: `TfidfVectorizer` → `MultinomialNB`.
- Stratified train/test split, fixed random seed.
- Headline metric: **F1 on the spam class** (not raw accuracy — a model predicting "ham" always would score ~87% accuracy while being useless). Also report spam-class precision/recall, and discuss the cost asymmetry (false positives block real messages; false negatives let spam through).
- Output: `baseline_model.joblib`, `metrics_report.md`, confusion matrix plot.

**Iteration (`spam-harm/iterate/`):**
- Try `LogisticRegression` on the same TF-IDF features, tuned via grid search (regularization strength `C`, n-gram range 1-2).
- Compare against baseline honestly — report what changed overall AND per-class, including any regressions, not just the headline number (matching `stage03`'s convention).
- Test whether stopword removal / stemming variants measurably help or are noise; document what's kept vs. discarded and why.
- Final model selection documented in `model_choice.md`, based on spam-class F1 with tradeoffs stated plainly.

## Packaging + deployment

**`spam-harm/app/predict.py`:**
- Wraps the final model + fitted vectorizer into `predict_message(text: str) -> dict` returning `{"label": "spam"/"ham", "confidence": float, "probabilities": {...}}`.
- Vectorizer only ever `.transform()`s at inference, never re-fits.
- Explicit input validation (empty string, non-string input, extremely long input) informed directly by the silent-wrong-answer bugs found in the Disney+ project's `stress_test.py` (e.g. bad input types must raise loudly, not silently produce a bogus prediction).

**`spam-harm/app/app.py` (Streamlit):**
- Single text box → "Classify" button → predicted label, confidence. Per-word TF-IDF contribution display is an optional nice-to-have, not required for this to be considered done.
- Low-confidence warning banner, same pattern as the Disney+ app.

**`spam-harm/app/stress_test.py`:** adversarial/edge-case inputs — empty string, whitespace-only, extremely long text, non-English/emoji-only text, `None`/wrong-type input. Same "silent-wrong-answer bugs are worse than crashes" philosophy as the Disney+ project.

**Deployment:** a new, separate Streamlit Community Cloud app (own URL, independent from the Disney+ demo). `spam-harm/app/DEPLOYMENT.md` follows the same structure as the Disney+ one, but bakes in two lessons learned this session up front:
1. Pin `requirements.txt` to versions actually tested locally.
2. Always re-verify the live app against the *exact* documented smoke-test input before suspecting the environment — a different input legitimately producing a different prediction is not a bug.

## Verification approach

No new test framework introduced — follows the convention already established in this repo:
- Each phase script gets a `__main__`-style smoke test (run directly, confirm expected output on a known example), matching `predict.py`'s existing pattern.
- `stress_test.py` is the adversarial/edge-case suite, same lightweight custom-runner style as `stage04/stress_test.py` (no pytest).
- Deployment is not considered "done" until the live Streamlit app is re-verified against the exact documented smoke-test input.

## Out of scope (this spec)

- IMDB sentiment analysis — separate future spec, own spec → plan → implementation cycle.
- Any change to the Disney+ (`stage01`-`stage04`) project.
- Advanced NLP (embeddings, transformers) — README specifies classical TF-IDF + Naive Bayes/Logistic Regression, consistent with "NLP-lite."
