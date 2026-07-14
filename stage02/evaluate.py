"""
stage02/evaluate.py

Week 2, Day 9: Measuring performance

Evaluates the saved baseline model against the held-out test set.
Reports accuracy, macro F1, weighted F1, a full per-class breakdown, and
a confusion matrix -- then compares macro F1 directly against the
naive-baseline number from Day 6's problem_statement.md (0.040), since
that's the whole point of having set it: a number to actually beat.

Run:
    python evaluate.py

Produces:
    stage02/metrics_report.md
    stage02/confusion_matrix.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "features"
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"
REPORT_PATH = SCRIPT_DIR / "metrics_report.md"
CM_PATH = SCRIPT_DIR / "confusion_matrix.png"

# From Day 6's problem_statement.md -- naive majority-class baseline
NAIVE_BASELINE_MACRO_F1 = 0.040


def main():
    X_test = np.load(FEATURES_DIR / "X_test.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy", allow_pickle=True)
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("=" * 60)
    print("BASELINE MODEL — TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:           {acc:.4f} ({acc*100:.1f}%)")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")
    print(f"Balanced accuracy:  {bal_acc:.4f}")
    print()
    print(f"Naive baseline (Day 6, always predict majority class): macro F1 = {NAIVE_BASELINE_MACRO_F1:.4f}")
    if macro_f1 > NAIVE_BASELINE_MACRO_F1:
        multiplier = macro_f1 / NAIVE_BASELINE_MACRO_F1
        print(f"RESULT: beats the naive baseline ({macro_f1:.4f} vs {NAIVE_BASELINE_MACRO_F1:.4f} -- {multiplier:.1f}x)")
    else:
        print(f"RESULT: does NOT beat the naive baseline -- worth investigating why before moving on")
    print()
    print("Per-class breakdown:")
    print(report)

    # --- confusion matrix ---
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Baseline model — confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=120)
    print(f"\nSaved confusion matrix -> {CM_PATH}")

    # --- metrics report ---
    with open(REPORT_PATH, "w") as f:
        f.write("# Baseline Model — Metrics Report\n\n")
        f.write(f"- **Accuracy:** {acc:.4f} ({acc*100:.1f}%)\n")
        f.write(f"- **Macro F1:** {macro_f1:.4f}\n")
        f.write(f"- **Weighted F1:** {weighted_f1:.4f}\n")
        f.write(f"- **Balanced accuracy:** {bal_acc:.4f}\n\n")
        f.write(f"**Naive baseline (Day 6):** macro F1 = {NAIVE_BASELINE_MACRO_F1:.4f}\n\n")
        if macro_f1 > NAIVE_BASELINE_MACRO_F1:
            f.write(f"**Result:** beats the naive baseline by {macro_f1/NAIVE_BASELINE_MACRO_F1:.1f}x.\n\n")
        else:
            f.write("**Result:** does NOT beat the naive baseline.\n\n")
        f.write("## Per-class breakdown\n\n```\n")
        f.write(report)
        f.write("\n```\n\n")
        f.write("![Confusion matrix](confusion_matrix.png)\n")
    print(f"Saved metrics report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
