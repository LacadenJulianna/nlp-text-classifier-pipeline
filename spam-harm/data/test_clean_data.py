"""
spam-harm/data/test_clean_data.py

Sanity checks for clean_data.py. Run after any change to the cleaning
pipeline to catch a silent regression.

Run:
    python test_clean_data.py
"""

import pandas as pd
from pathlib import Path
from load_raw import load_raw_data, RAW_PATH
from clean_data import clean_data

SCRIPT_DIR = Path(__file__).resolve().parent


def run_checks():
    raw = load_raw_data(RAW_PATH)
    clean = clean_data(raw)

    checks = [
        ("no missing message", clean["message"].isnull().sum() == 0),
        ("no missing label", clean["label"].isnull().sum() == 0),
        ("label is 0/1 int, not string", set(clean["label"].unique()) <= {0, 1}),
        ("no duplicate messages remain", clean["message"].duplicated().sum() == 0),
        ("no empty-string messages", (clean["message"].str.strip() == "").sum() == 0),
        ("spam is the minority class",
         clean["label"].mean() < 0.5),
    ]

    print(f"Running {len(checks)} checks...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
