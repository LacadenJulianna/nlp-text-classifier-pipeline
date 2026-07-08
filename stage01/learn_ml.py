"""
stage01/learn-ml.py

Week 1, Day 2: Python for data + getting your dataset
Project: Predict Disney+ content rating (TV-G, PG, TV-14, ...) from
genre, runtime, release year, and country.

This file is today's pandas practice only: load, filter, groupby.
Day 3's cleaning pipeline lives separately in clean_data.py, since that's
a reusable artifact later stages import -- not exploratory scratch work.
"""

import pandas as pd
from pathlib import Path

# Raw data lives at the project root, one level up from stage01/
# (confirmed from the actual repo layout -- adjust here if that changes).
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR.parent / "disney_plus_titles.csv"

# ---------------------------------------------------------------------------
# DAY 2: Python for data + getting your dataset
# ---------------------------------------------------------------------------
# Goal for today: load pandas, and get comfortable with load / filter /
# groupby on your ACTUAL dataset, not a toy example.

def day2_load_and_explore():
    df = pd.read_csv(RAW_PATH)

    print("=" * 60)
    print("DAY 2: Load + explore")
    print("=" * 60)

    # --- load ---
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns and types:\n{df.dtypes}")

    # --- filter ---
    # Example: how many titles came out in the last 5 years of the catalog?
    recent = df[df["release_year"] >= 2017]
    print(f"\nTitles released 2017 or later: {len(recent)} of {len(df)}")

    # Example: isolate just the TV shows
    tv_shows = df[df["type"] == "TV Show"]
    print(f"TV Shows: {len(tv_shows)} | Movies: {len(df) - len(tv_shows)}")

    # --- groupby ---
    # How many titles per rating category?
    by_rating = df.groupby("rating", dropna=False)["title"].count().sort_values(ascending=False)
    print(f"\nTitle count by rating:\n{by_rating}")

    # Average release year per content type -- do TV shows skew newer?
    avg_year_by_type = df.groupby("type")["release_year"].mean()
    print(f"\nAverage release year by type:\n{avg_year_by_type}")

    return df


if __name__ == "__main__":
    day2_load_and_explore()
