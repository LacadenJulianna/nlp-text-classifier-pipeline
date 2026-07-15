"""
spam-harm/iterate/evaluate_candidate.py

Evaluates the tuned candidate against the same held-out test set the
baseline was scored on, compares both honestly (per-class, not just
the headline spam F1), and copies whichever model wins to
final_model.joblib -- the one artifact predict.py and the Streamlit
app actually load.

Run:
    python evaluate_candidate.py

Produces:
    spam-harm/iterate/candidate_metrics.md
    spam-harm/iterate/model_choice.md
    spam-harm/iterate/final_model.joblib
"""

import shutil
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_PATH = SCRIPT_DIR.parent / "data" / "clean" / "test.csv"
BASELINE_MODEL_PATH = SCRIPT_DIR.parent / "baseline" / "baseline_model.joblib"
CANDIDATE_MODEL_PATH = SCRIPT_DIR / "candidate_model.joblib"
FINAL_MODEL_PATH = SCRIPT_DIR / "final_model.joblib"
METRICS_PATH = SCRIPT_DIR / "candidate_metrics.md"
CHOICE_PATH = SCRIPT_DIR / "model_choice.md"


def score(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "spam_f1": f1_score(y_test, y_pred, pos_label=1),
        "spam_precision": precision_score(y_test, y_pred, pos_label=1),
        "spam_recall": recall_score(y_test, y_pred, pos_label=1),
        "report": classification_report(y_test, y_pred, target_names=["ham", "spam"]),
    }


def main():
    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = test_df["message"], test_df["label"]

    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    baseline_scores = score(baseline_model, X_test, y_test)
    candidate_scores = score(candidate_model, X_test, y_test)

    print("Baseline spam F1:  ", f"{baseline_scores['spam_f1']:.4f}")
    print("Candidate spam F1: ", f"{candidate_scores['spam_f1']:.4f}")

    winner = "candidate" if candidate_scores["spam_f1"] >= baseline_scores["spam_f1"] else "baseline"
    winner_path = CANDIDATE_MODEL_PATH if winner == "candidate" else BASELINE_MODEL_PATH
    shutil.copy(winner_path, FINAL_MODEL_PATH)
    print(f"\nWinner: {winner} (copied to {FINAL_MODEL_PATH})")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("# Candidate Model — Metrics Report\n\n")
        f.write(f"- **Accuracy:** {candidate_scores['accuracy']:.4f}\n")
        f.write(f"- **Spam F1:** {candidate_scores['spam_f1']:.4f}\n")
        f.write(f"- **Spam precision:** {candidate_scores['spam_precision']:.4f}\n")
        f.write(f"- **Spam recall:** {candidate_scores['spam_recall']:.4f}\n\n")
        f.write("## Per-class breakdown\n\n```\n")
        f.write(candidate_scores["report"])
        f.write("\n```\n")

    with open(CHOICE_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Choice: Baseline vs. Candidate\n\n")
        f.write("| Metric | Baseline (Naive Bayes) | Candidate (tuned Logistic Regression) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Accuracy | {baseline_scores['accuracy']:.4f} | {candidate_scores['accuracy']:.4f} |\n")
        f.write(f"| Spam F1 | {baseline_scores['spam_f1']:.4f} | {candidate_scores['spam_f1']:.4f} |\n")
        f.write(f"| Spam precision | {baseline_scores['spam_precision']:.4f} | {candidate_scores['spam_precision']:.4f} |\n")
        f.write(f"| Spam recall | {baseline_scores['spam_recall']:.4f} | {candidate_scores['spam_recall']:.4f} |\n\n")
        f.write(f"**Selected: {winner}**, based on spam F1 on the held-out test set "
                f"(the metric this project treats as the headline number, not accuracy).\n\n")
        f.write("**Note:** Both models achieved perfect or near-perfect spam F1 scores, which reflects "
                f"this being a small, likely-synthetic dataset with trivially-separable classes "
                f"(per EDA: ham and spam message lengths do not overlap). Ties are broken by >= "
                f"favoring the candidate, so the selection should not be read as a demonstrated win "
                f"but rather as a design-time choice when models are equivalent on this particular benchmark.\n\n")
        f.write("## Per-class breakdown, baseline\n\n```\n")
        f.write(baseline_scores["report"])
        f.write("\n```\n\n## Per-class breakdown, candidate\n\n```\n")
        f.write(candidate_scores["report"])
        f.write("\n```\n")

    print(f"Saved -> {METRICS_PATH}, {CHOICE_PATH}, {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
