"""
spam-harm/baseline/evaluate.py

Evaluates the baseline model against the held-out test set. Headline
metric is spam-class F1, not accuracy -- a model that always predicts
"ham" would score accuracy equal to the ham proportion of the test set
while catching zero spam, which is exactly why accuracy alone is the
wrong number to lead with here. The naive baseline below makes that
concrete: it's the "always predict ham" F1, which is 0.0 by
construction (zero recall on the spam class). The ham proportion is
computed from the actual test set below rather than hard-coded, since
it depends on which raw dataset is in use.

Run:
    python evaluate.py

Produces:
    spam-harm/baseline/metrics_report.md
    spam-harm/baseline/confusion_matrix.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_PATH = SCRIPT_DIR.parent / "data" / "clean" / "test.csv"
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"
REPORT_PATH = SCRIPT_DIR / "metrics_report.md"
CM_PATH = SCRIPT_DIR / "confusion_matrix.png"

# "Always predict ham" -- zero recall on spam, so spam F1 = 0.0 by
# construction. The number to beat.
NAIVE_BASELINE_SPAM_F1 = 0.0


def main():
    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    y_test = test_df["label"]
    y_pred = model.predict(test_df["message"])
    ham_proportion = (y_test == 0).mean()

    acc = accuracy_score(y_test, y_pred)
    spam_f1 = f1_score(y_test, y_pred, pos_label=1)
    spam_precision = precision_score(y_test, y_pred, pos_label=1)
    spam_recall = recall_score(y_test, y_pred, pos_label=1)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

    print("=" * 60)
    print("BASELINE MODEL — TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:       {acc:.4f} ({acc*100:.1f}%)")
    print(f"Spam F1:        {spam_f1:.4f}")
    print(f"Spam precision: {spam_precision:.4f}  (of messages flagged spam, how many really are)")
    print(f"Spam recall:    {spam_recall:.4f}  (of real spam, how much got caught)")
    print(f"\nNaive baseline (always predict ham): spam F1 = {NAIVE_BASELINE_SPAM_F1:.4f}")
    print()
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Baseline model — confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=120)
    print(f"\nSaved confusion matrix -> {CM_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Baseline Model — Metrics Report\n\n")
        f.write(f"- **Accuracy:** {acc:.4f} ({acc*100:.1f}%)\n")
        f.write(f"- **Spam F1:** {spam_f1:.4f}\n")
        f.write(f"- **Spam precision:** {spam_precision:.4f}\n")
        f.write(f"- **Spam recall:** {spam_recall:.4f}\n\n")
        f.write(f"**Naive baseline** (always predict ham): spam F1 = {NAIVE_BASELINE_SPAM_F1:.4f}\n\n")
        f.write(f"Accuracy alone is misleading here: this test set is {ham_proportion*100:.1f}% ham, so a "
                f"model that always predicts \"ham\" would score {ham_proportion*100:.1f}% accuracy while "
                "catching zero spam. Spam F1 is the number that actually reflects "
                "whether the model is useful.\n\n")
        f.write("## Per-class breakdown\n\n```\n")
        f.write(report)
        f.write("\n```\n\n")
        f.write("![Confusion matrix](confusion_matrix.png)\n")
    print(f"Saved metrics report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
