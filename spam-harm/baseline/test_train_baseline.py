"""
spam-harm/baseline/test_train_baseline.py

Confirms the baseline pipeline trains and produces a usable model --
not a real evaluation (see evaluate.py for that), just a sanity check
that the artifact is a working Pipeline.

Run:
    python test_train_baseline.py
"""

import subprocess
import sys
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "train_baseline.py")], check=True)

    model = joblib.load(MODEL_PATH)
    pred = model.predict(["free entry to win a prize call now"])
    proba = model.predict_proba(["hey are we still on for lunch"])

    checks = [
        ("model file was created", MODEL_PATH.exists()),
        ("model has predict()", hasattr(model, "predict")),
        ("model has predict_proba()", hasattr(model, "predict_proba")),
        ("predict returns one label per input", len(pred) == 1),
        ("predict_proba returns two class probabilities", proba.shape[1] == 2),
        ("probabilities sum to ~1.0", abs(proba[0].sum() - 1.0) < 1e-6),
    ]

    print(f"\nRunning {len(checks)} checks...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
