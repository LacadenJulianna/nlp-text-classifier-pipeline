"""
stage03/train_candidate.py

Week 3, Day 12: Train it

Trains the Day 11 candidate (Random Forest, default settings -- see
model_choice.md for why class_weight="balanced" was tested and rejected)
and compares it directly against the Week 2 baseline on the same metric,
using the EXACT same train/test split (reuses stage02's saved arrays
rather than re-deriving the split, so the comparison is apples-to-apples).

Run:
    python train_candidate.py
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE02_FEATURES = SCRIPT_DIR.parent / "stage02" / "features"
MODEL_PATH = SCRIPT_DIR / "candidate_model.joblib"
LOG_PATH = SCRIPT_DIR / "candidate_metrics.md"

# From stage02/metrics_report.md
BASELINE_MACRO_F1 = 0.5272
BASELINE_ACCURACY = 0.6172

RANDOM_STATE = 42
N_ESTIMATORS = 300


def main():
    X_train = np.load(STAGE02_FEATURES / "X_train.npy")
    X_test = np.load(STAGE02_FEATURES / "X_test.npy")
    y_train = np.load(STAGE02_FEATURES / "y_train.npy", allow_pickle=True)
    y_test = np.load(STAGE02_FEATURES / "y_test.npy", allow_pickle=True)

    print(f"Training Random Forest on {X_train.shape[0]} rows, {X_train.shape[1]} features "
          f"(reusing Week 2's exact split for a fair comparison)")

    model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nCandidate (Random Forest): macro F1 = {macro_f1:.4f}, accuracy = {acc:.4f}")
    print(f"Baseline  (Logistic Reg.):  macro F1 = {BASELINE_MACRO_F1:.4f}, accuracy = {BASELINE_ACCURACY:.4f}")
    delta = macro_f1 - BASELINE_MACRO_F1
    print(f"Delta: {delta:+.4f} macro F1 ({delta/BASELINE_MACRO_F1*100:+.1f}% relative)")
    print()
    print(report)

    joblib.dump(model, MODEL_PATH)
    with open(LOG_PATH, "w") as f:
        f.write("# Day 12 — Candidate Model Metrics\n\n")
        f.write("| | Macro F1 | Accuracy |\n|---|---|---|\n")
        f.write(f"| Baseline (Week 2, logistic regression) | {BASELINE_MACRO_F1:.4f} | {BASELINE_ACCURACY:.4f} |\n")
        f.write(f"| Candidate (Random Forest) | {macro_f1:.4f} | {acc:.4f} |\n")
        f.write(f"| **Delta** | **{delta:+.4f} ({delta/BASELINE_MACRO_F1*100:+.1f}%)** | {acc-BASELINE_ACCURACY:+.4f} |\n\n")
        f.write("## Per-class breakdown\n\n```\n" + report + "\n```\n")

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved metrics log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
