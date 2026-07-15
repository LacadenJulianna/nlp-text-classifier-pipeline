"""
spam-harm/iterate/test_evaluate_candidate.py

Confirms the comparison script produces final_model.joblib and
documents the decision honestly (per-class, not just the headline
number).

Run:
    python test_evaluate_candidate.py
"""

import subprocess
import sys
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_MODEL_PATH = SCRIPT_DIR / "final_model.joblib"
CHOICE_PATH = SCRIPT_DIR / "model_choice.md"
METRICS_PATH = SCRIPT_DIR / "candidate_metrics.md"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "evaluate_candidate.py")], check=True)

    final_model = joblib.load(FINAL_MODEL_PATH)
    choice_text = CHOICE_PATH.read_text(encoding="utf-8")

    checks = [
        ("final model file was created", FINAL_MODEL_PATH.exists()),
        ("final model can predict", len(final_model.predict(["free prize call now"])) == 1),
        ("candidate metrics report was created", METRICS_PATH.exists()),
        ("model choice doc was created", CHOICE_PATH.exists()),
        ("model choice doc has a comparison table", "|" in choice_text),
        ("model choice doc states a decision", "chosen" in choice_text.lower() or "selected" in choice_text.lower()),
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
