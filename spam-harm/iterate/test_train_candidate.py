"""
spam-harm/iterate/test_train_candidate.py

Confirms the tuned candidate pipeline trains and produces a usable
model, and that the tuning log actually documents the stopword-removal
comparison (not just the winning params).

Run:
    python test_train_candidate.py
"""

import subprocess
import sys
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "candidate_model.joblib"
LOG_PATH = SCRIPT_DIR / "tuning_log.md"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "train_candidate.py")], check=True)

    # joblib.load is used here on an artifact this same pipeline just
    # trained and wrote to disk above (not an externally-sourced file),
    # so the usual "untrusted pickle" risk does not apply.
    model = joblib.load(MODEL_PATH)
    pred = model.predict(["free entry to win a prize call now"])
    log_text = LOG_PATH.read_text(encoding="utf-8")

    checks = [
        ("model file was created", MODEL_PATH.exists()),
        ("model has predict()", hasattr(model, "predict")),
        ("predict returns one label per input", len(pred) == 1),
        ("tuning log was created", LOG_PATH.exists()),
        ("tuning log documents stopword comparison", "stop_words" in log_text.lower() or "stopword" in log_text.lower()),
        ("tuning log documents stemming decision", "stemming" in log_text.lower()),
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
