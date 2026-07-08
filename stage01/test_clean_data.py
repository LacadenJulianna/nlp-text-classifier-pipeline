"""
stage01/test_clean_data.py

Sanity checks for clean_data.py. Run this after any change to the
cleaning pipeline to catch a silent regression, instead of re-reading
printed row counts by eye every time.

Run:
    python test_clean_data.py
"""

import pandas as pd
from pathlib import Path
from clean_data import clean_data

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR.parent / "disney_plus_titles.csv"


def run_checks():
    raw = pd.read_csv(RAW_PATH)
    clean = clean_data(raw)

    checks = [
        ("no missing `rating`", clean["rating"].isnull().sum() == 0),
        ("no missing `country`", clean["country"].isnull().sum() == 0),
        ("duration_value fully populated", clean["duration_value"].isnull().sum() == 0),
        ("duration_unit only has min/Season",
         set(clean["duration_unit"].unique()) <= {"min", "Season"}),
        ("no duplicate rows remain",
         clean.drop(columns=["genre_list"]).duplicated().sum() == 0),
        (f"dropped exactly {raw['rating'].isnull().sum()} rows (missing rating)",
         (len(raw) - len(clean)) == raw["rating"].isnull().sum()),
        ("genre_list is a list, not a raw string",
         isinstance(clean["genre_list"].iloc[0], list)),
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