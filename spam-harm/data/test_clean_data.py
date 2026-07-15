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
TRAIN_PATH = SCRIPT_DIR / "clean" / "train.csv"
TEST_PATH = SCRIPT_DIR / "clean" / "test.csv"


def run_checks():
    raw = load_raw_data(RAW_PATH)
    clean = clean_data(raw)

    # Check split output files existence
    train_exists = TRAIN_PATH.exists()
    test_exists = TEST_PATH.exists()

    # Check split output file headers and content
    train_df = None
    test_df = None
    train_header_ok = False
    test_header_ok = False
    no_leakage = False
    row_count_sum_ok = False

    if train_exists and test_exists:
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        train_header_ok = list(train_df.columns) == ["label", "message"]
        test_header_ok = list(test_df.columns) == ["label", "message"]

        # Check for data leakage (no overlap in messages)
        train_messages = set(train_df["message"])
        test_messages = set(test_df["message"])
        no_leakage = len(train_messages & test_messages) == 0

        # Check row count sum matches cleaned data
        row_count_sum_ok = len(train_df) + len(test_df) == len(clean)

    checks = [
        ("no missing message", clean["message"].isnull().sum() == 0),
        ("no missing label", clean["label"].isnull().sum() == 0),
        ("label is 0/1 int, not string", set(clean["label"].unique()) <= {0, 1}),
        ("no duplicate messages remain", clean["message"].duplicated().sum() == 0),
        ("no empty-string messages", (clean["message"].str.strip() == "").sum() == 0),
        ("spam is the minority class",
         clean["label"].mean() < 0.5),
        ("train.csv exists", train_exists),
        ("test.csv exists", test_exists),
        ("train.csv has correct header", train_header_ok),
        ("test.csv has correct header", test_header_ok),
        ("no data leakage (train/test overlap)", no_leakage),
        ("train + test row count equals cleaned data", row_count_sum_ok),
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
