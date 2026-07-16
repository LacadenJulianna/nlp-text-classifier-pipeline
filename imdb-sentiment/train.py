"""Train and evaluate the IMDB review sentiment classifier.

Single-pass pipeline (no separate baseline/candidate stages): TF-IDF +
Logistic Regression is already the well-known strong classical baseline for
this dataset, so we tune it directly instead of iterating through a weaker
model first.
"""
import re
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "raw" / "IMDB_Dataset.csv"
MODEL_PATH = ROOT / "final_model.joblib"
METRICS_PATH = ROOT / "metrics_report.md"

HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean_review(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["review", "sentiment"])
    df["review"] = df["review"].astype(str).map(clean_review)
    df = df[df["review"].str.len() > 0]
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=5,
            max_features=30000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(C=10, max_iter=1000, solver="liblinear")),
    ])


def main():
    print("Loading and cleaning data...")
    df = load_data()
    print(f"{len(df)} rows after cleaning, class balance:\n{df['sentiment'].value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, stratify=df["sentiment"], random_state=42
    )

    pipeline = build_pipeline()
    print("Training...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Trained in {train_time:.1f}s")

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

    print(f"Accuracy: {acc:.4f}")
    print(report)

    joblib.dump(pipeline, MODEL_PATH)

    METRICS_PATH.write_text(
        f"""# IMDB Sentiment Classifier — Metrics

Model: TF-IDF (unigrams+bigrams, max_features=30000, min_df=5, sublinear_tf) +
Logistic Regression (C=10, liblinear).

Train/test split: 80/20 stratified, random_state=42.
Train size: {len(X_train)}, test size: {len(X_test)}.
Training time: {train_time:.1f}s.

**Accuracy: {acc:.4f}**

## Classification report

```
{report}
```

## Confusion matrix (rows=actual, cols=predicted; order: negative, positive)

```
{cm}
```
""",
        encoding="utf-8",
    )
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
