"""
spam-harm/baseline/test_evaluate.py

Confirms evaluate.py produces the expected output files with the
expected structure.

Run:
    python test_evaluate.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR / "metrics_report.md"
CM_PATH = SCRIPT_DIR / "confusion_matrix.png"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "evaluate.py")], check=True)

    report_text = REPORT_PATH.read_text()

    checks = [
        ("metrics report was created", REPORT_PATH.exists()),
        ("confusion matrix image was created", CM_PATH.exists()),
        ("report mentions spam F1", "Spam F1" in report_text or "spam F1" in report_text.lower()),
        ("report mentions naive baseline comparison", "naive" in report_text.lower()),
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
