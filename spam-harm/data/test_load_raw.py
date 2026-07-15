"""
spam-harm/data/test_load_raw.py

Sanity checks for load_raw.py. Run after any change to confirm the loader
still produces the expected two-column shape, regardless of which raw
distribution (Kaggle CSV or plain label/message CSV) is on disk.

Run:
    python test_load_raw.py
"""

import pandas as pd
from pathlib import Path
from load_raw import load_raw_data

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / "raw" / "spam.csv"


def run_checks():
    if not RAW_PATH.exists():
        raise SystemExit(
            f"Raw dataset not found at {RAW_PATH}. Place the raw dataset CSV "
            "there before running this test (see Task 1's dataset note in "
            "the implementation plan)."
        )

    df = load_raw_data(RAW_PATH)

    checks = [
        ("has exactly two columns", list(df.columns) == ["label", "message"]),
        ("has at least 900 rows", len(df) >= 900),
        ("label column only has spam/ham values",
         set(df["label"].str.lower().unique()) <= {"spam", "ham"}),
        ("message column has no nulls", df["message"].isnull().sum() == 0),
        ("both spam and ham present", set(df["label"].str.lower().unique()) == {"spam", "ham"}),
    ]

    print(f"Running {len(checks)} checks against {RAW_PATH}...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
