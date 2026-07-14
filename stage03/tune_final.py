"""
stage03/tune_final.py

Week 3, Day 14: Tune it + lock it in

Light tuning only, as the roadmap asks -- a small grid, not exhaustive
search. Uses Day 12's feature set unchanged (Day 13 decided against
both candidate tweaks). Tuning happens via cross-validation on the
TRAINING set only; the test set is touched exactly once at the end, to
report a final, honest number -- not used to pick hyperparameters.

Small CV fold count (3) is deliberate: the smallest class has only 10
training examples, so more folds would leave very few (or zero) of that
class per fold.

Run:
    python tune_final.py
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE02_FEATURES = SCRIPT_DIR.parent / "stage02" / "features"
FINAL_MODEL_PATH = SCRIPT_DIR / "final_model.joblib"
LOG_PATH = SCRIPT_DIR / "tuning_log.md"

RANDOM_STATE = 42

# Small, non-exhaustive grid -- 3 x 3 x 2 = 18 combinations
PARAM_GRID = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 15, 25],
    "min_samples_leaf": [1, 2],
}


def main():
    X_train = np.load(STAGE02_FEATURES / "X_train.npy")
    X_test = np.load(STAGE02_FEATURES / "X_test.npy")
    y_train = np.load(STAGE02_FEATURES / "y_train.npy", allow_pickle=True)
    y_test = np.load(STAGE02_FEATURES / "y_test.npy", allow_pickle=True)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        PARAM_GRID,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    print(f"Running grid search: {len(PARAM_GRID['n_estimators']) * len(PARAM_GRID['max_depth']) * len(PARAM_GRID['min_samples_leaf'])} combinations, 3-fold CV, training data only...")
    search.fit(X_train, y_train)

    print(f"\nBest CV macro F1: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")

    # Final, one-time check against the held-out test set
    final_model = search.best_estimator_
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nFinal model on held-out test set: macro F1 = {macro_f1:.4f}, accuracy = {acc:.4f}")
    print(f"(Day 12 untuned candidate was: macro F1 = 0.5757)")
    print()
    print(report)

    joblib.dump(final_model, FINAL_MODEL_PATH)
    with open(LOG_PATH, "w") as f:
        f.write("# Day 14 — Tuning Log\n\n")
        f.write(f"Grid: `{PARAM_GRID}`\n\n")
        f.write(f"Best CV macro F1 (training data, 3-fold): {search.best_score_:.4f}\n\n")
        f.write(f"Best params: `{search.best_params_}`\n\n")
        f.write(f"## Final test-set result\n\n")
        f.write(f"- Macro F1: {macro_f1:.4f}\n- Accuracy: {acc:.4f}\n\n")
        f.write("```\n" + report + "\n```\n")

    print(f"\nSaved final model -> {FINAL_MODEL_PATH}")
    print(f"Saved tuning log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
