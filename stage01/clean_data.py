"""
stage01/clean_data.py

Week 1, Day 3: Data cleaning

Standalone, reusable cleaning pipeline for the Disney+ titles dataset.
Later stages (Day 4 EDA, model training, etc.) should import
`clean_data()` from here rather than re-running exploratory code --
this script's only job is turning the raw CSV into a clean one.

Run directly:
    python clean_data.py
    python clean_data.py --input other_raw.csv --output other_clean.csv

Or import:
    from clean_data import clean_data
    df = clean_data(pd.read_csv("disney_plus_titles.csv"))
"""

import argparse
import pandas as pd
from pathlib import Path

# Raw data lives at the project root, one level up from stage01/
# (confirmed from the actual repo layout -- adjust here if that changes).
# The cleaned output stays inside stage01/, since it's this stage's own
# artifact -- stage02 can reference it via a relative path from there.
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR.parent / "disney_plus_titles.csv"
CLEAN_PATH = SCRIPT_DIR / "disney_plus_titles_clean.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Disney+ titles dataframe.

    Decisions made here (full "why" for each lives in workflow.md):
      1. Drop rows with missing `rating` -- that's the prediction target;
         a row with no label can't be trained on or scored against.
      2. Leave `director` / `cast` columns in place but don't engineer
         features from them (33% / 13% missing, high-cardinality).
      3. Fill missing `country` with "Unknown" rather than dropping rows
         -- country is a feature we want to keep.
      4. Split `duration` into `duration_value` (int) + `duration_unit`
         (min/Season) -- the raw column mixes two different units
         depending on `type`.
      5. Split `listed_in` into `genre_list` -- one title can carry
         multiple genres.
      6. Parse `date_added` to a real datetime, where present.

    Known limitation, not fixed here: `duration_unit` perfectly predicts
    `type` (every Movie is "min", every TV Show is "Season"). If a later
    stage predicts `type` instead of `rating`, drop `duration` from the
    feature set first -- otherwise the model is just reading a
    disguised copy of the label.
    """
    df = df.copy()
    start_rows = len(df)

    # duplicates
    dupes = df.duplicated().sum()
    df = df.drop_duplicates()

    # missing values
    df = df.dropna(subset=["rating"])
    df["country"] = df["country"].fillna("Unknown")
    # date_added's 3 missing values are left as-is (not used as a
    # feature this week) rather than inventing a fake date.

    # mismatched types
    duration_split = df["duration"].str.extract(
        r"(?P<duration_value>\d+)\s*(?P<duration_unit>\w+)"
    )
    df["duration_value"] = pd.to_numeric(duration_split["duration_value"])
    df["duration_unit"] = duration_split["duration_unit"].replace(
        {"Season": "Season", "Seasons": "Season"}  # normalize singular/plural
    )
    df["genre_list"] = df["listed_in"].str.split(", ")
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")

    dropped = start_rows - len(df)
    print(f"Duplicate rows found: {dupes}")
    print(f"Rows: {start_rows} -> {len(df)} ({dropped} dropped, missing `rating`)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Clean the Disney+ titles dataset.")
    parser.add_argument("--input", default=RAW_PATH, help="Path to raw CSV")
    parser.add_argument("--output", default=CLEAN_PATH, help="Path to write cleaned CSV")
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    clean_df = clean_data(raw_df)
    clean_df.to_csv(args.output, index=False)
    print(f"Saved -> {args.output} ({len(clean_df)} rows)")


if __name__ == "__main__":
    main()