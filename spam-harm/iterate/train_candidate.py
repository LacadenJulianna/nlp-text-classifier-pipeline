"""
spam-harm/iterate/train_candidate.py

Trains a tuned Logistic Regression candidate on the same TF-IDF
features as the baseline, via grid search over:
  - clf__C: regularization strength
  - tfidf__ngram_range: unigrams only vs. unigrams+bigrams
  - tfidf__stop_words: None vs. "english" -- this is how the "does
    stopword removal help" question gets answered, as a natural
    byproduct of the same search rather than a separate experiment.

Stemming was considered (per the design spec) but not implemented: it
needs an extra dependency (nltk + a corpus download) that doesn't pay
for itself on a "NLP-lite" project this size. Documented here, not
silently dropped.

Run:
    python train_candidate.py
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR.parent / "data" / "clean" / "train.csv"
MODEL_PATH = SCRIPT_DIR / "candidate_model.joblib"
LOG_PATH = SCRIPT_DIR / "tuning_log.md"

RANDOM_STATE = 42

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__stop_words": [None, "english"],
    "clf__C": [0.01, 0.1, 1, 10],
}


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Training on {len(train_df)} rows")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, PARAM_GRID, scoring="f1", cv=cv, n_jobs=-1)
    grid.fit(train_df["message"], train_df["label"])

    print(f"Best CV spam F1: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

    # Isolate the stopword-removal effect specifically: mean CV score
    # for stop_words=None vs. "english", averaged across every other
    # param combination, so the comparison isn't just "the single best
    # row happened to use X."
    results = pd.DataFrame(grid.cv_results_)
    stopword_effect = results.groupby(
        results["params"].apply(lambda p: p["tfidf__stop_words"])
    )["mean_test_score"].mean()
    print(f"\nMean CV spam F1 by stop_words setting:\n{stopword_effect}")

    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("# Candidate Model — Tuning Log\n\n")
        f.write("## Grid searched\n\n")
        f.write(f"```\n{PARAM_GRID}\n```\n\n")
        f.write(f"## Best result\n\n")
        f.write(f"- **Best CV spam F1:** {grid.best_score_:.4f}\n")
        f.write(f"- **Best params:** {grid.best_params_}\n\n")
        f.write("## Stopword removal: does it help?\n\n")
        f.write(f"Mean CV spam F1 by `tfidf__stop_words` setting (averaged over all "
                f"other grid params, not just the single best row):\n\n")
        f.write(f"```\n{stopword_effect}\n```\n\n")
        best_stopword_setting = stopword_effect.idxmax()
        f.write(f"`{best_stopword_setting}` performs better on average -- "
                f"{'kept' if best_stopword_setting == grid.best_params_['tfidf__stop_words'] else 'note: differs from the single best combination, which used a different setting; the single-best row can differ from the on-average-better setting when it interacts with other params -- best_params_ is what actually gets used.'}\n\n")
        f.write("## Stemming: considered and discarded\n\n")
        f.write("Not implemented for this candidate. Stemming (e.g. via NLTK's "
                "PorterStemmer) would require an extra dependency plus a corpus "
                "download, which doesn't pay for itself on a dataset and project "
                "this size (\"NLP-lite\" per the project's own scope). Worth "
                "revisiting only if a future iteration shows TF-IDF token sparsity "
                "is actually a bottleneck.\n")
    print(f"Saved tuning log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
