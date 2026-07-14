"""
stage02/train_baseline.py

Week 2, Day 8: Your first model

Trains the simplest reasonable model for this task. Logistic regression,
not linear regression -- `rating` is categorical (Day 6's problem
statement), and linear regression is for continuous targets. A decision
tree was the other reasonable option; logistic regression was chosen as
the starting baseline since it's a well-understood default for
multi-class classification and gives interpretable coefficients.

This script trains and saves the model only. Full evaluation (accuracy,
macro F1, confusion matrix) is intentionally a separate step -- see
evaluate.py -- same reasoning as keeping clean_data.py and
test_clean_data.py separate in Stage 01: one script does the thing,
another checks how well it worked.

Run:
    python train_baseline.py
"""

import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
FEATURES_DIR = SCRIPT_DIR / "features"
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"

RANDOM_STATE = 42


def main():
    X_train = np.load(FEATURES_DIR / "X_train.npy")
    y_train = np.load(FEATURES_DIR / "y_train.npy", allow_pickle=True)

    print(f"Training on {X_train.shape[0]} rows, {X_train.shape[1]} features")

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        # no class_weight balancing here on purpose -- this is the
        # simplest reasonable baseline, not a tuned model. Class
        # imbalance handling is a documented candidate for the next
        # iteration, not this week's scope.
    )
    model.fit(X_train, y_train)

    # Rough sanity check only -- NOT the real evaluation. Training
    # accuracy tells you almost nothing about how the model performs on
    # data it hasn't seen; that's what evaluate.py (Day 9) is for.
    train_acc = accuracy_score(y_train, model.predict(X_train))
    print(f"Training-set accuracy (sanity check only, not a real metric): {train_acc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
