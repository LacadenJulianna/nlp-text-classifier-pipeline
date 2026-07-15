"""
spam-harm/baseline/train_baseline.py

Trains the simplest reasonable model for spam/ham text classification:
TF-IDF features + Multinomial Naive Bayes, the classic pairing for this
exact problem (fast, works well on short text, a well-understood
default before reaching for anything tuned).

The vectorizer and classifier are bundled into ONE sklearn Pipeline and
saved as a single artifact -- inference (predict.py) never needs to
separately manage a fitted vectorizer, it just calls .predict() on raw
text through the pipeline.

This script trains and saves the model only. Full evaluation is a
separate step -- see evaluate.py.

Run:
    python train_baseline.py
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR.parent / "data" / "clean" / "train.csv"
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"

RANDOM_STATE = 42


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Training on {len(train_df)} rows")

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ])
    model.fit(train_df["message"], train_df["label"])

    # Rough sanity check only -- NOT the real evaluation, which needs
    # the held-out test set (evaluate.py).
    train_pred = model.predict(train_df["message"])
    train_f1 = f1_score(train_df["label"], train_pred, pos_label=1)
    print(f"Training-set spam F1 (sanity check only, not a real metric): {train_f1:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
