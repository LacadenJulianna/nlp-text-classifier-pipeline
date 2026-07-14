"""
stage02/feature_pipeline.py

Week 2, Day 7: Preparing features

Turns the Stage 01 cleaned dataset into model-ready numeric arrays.

Order of operations matters here: the train/test split happens FIRST,
before anything data-derived is decided (which genres count as "top,"
the scaler's mean/std). Fitting those on the full dataset -- including
rows that end up in the test set -- would leak test-set information
into the feature definitions themselves. Same category of mistake as
the `type`/`duration` leakage flagged in Stage 01, just a different
place it can sneak in.

Run:
    python feature_pipeline.py

Produces (in stage02/features/):
    X_train.npy, X_test.npy, y_train.npy, y_test.npy
    feature_pipeline.joblib   -- fitted scaler + genre vocabulary, so any
                                  future data can be preprocessed identically
    feature_names.txt
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CLEAN_PATH = REPO_ROOT / "stage01" / "disney_plus_titles_clean.csv"
OUTPUT_DIR = SCRIPT_DIR / "features"

TOP_N_GENRES = 15
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_PATH)
    df["genre_list"] = df["genre_list"].apply(ast.literal_eval)
    return df


def encode_genres(genre_lists: pd.Series, vocab: list[str]) -> pd.DataFrame:
    """Multi-hot encode genres: one column per vocab genre, plus a
    genre_Other column for any genre outside the vocabulary. A title can
    have multiple genres, so this is multi-label, not one-hot."""
    vocab_set = set(vocab)
    rows = []
    for genres in genre_lists:
        row = {f"genre_{g}": int(g in genres) for g in vocab}
        row["genre_Other"] = int(any(g not in vocab_set for g in genres))
        rows.append(row)
    return pd.DataFrame(rows)


def build_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Two separate columns instead of one shared duration_value: minutes
    and seasons aren't the same unit, so scaling them together would mix
    two different scales into one number. Zero-filled where not
    applicable. Note: the zero/non-zero pattern across these two columns
    jointly reconstructs `type` perfectly (same leakage noted in Stage 01)
    -- fine for predicting `rating`, would need dropping if the target
    ever became `type`."""
    minutes = np.where(df["duration_unit"] == "min", df["duration_value"], 0)
    seasons = np.where(df["duration_unit"] == "Season", df["duration_value"], 0)
    return pd.DataFrame({"duration_minutes": minutes, "duration_seasons": seasons})


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df)} rows from {CLEAN_PATH}")

    y = df["rating"].values

    # --- split FIRST, stratified since some classes are tiny ---
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]
    y_train, y_test = df_train["rating"].values, df_test["rating"].values

    print(f"\nTrain: {len(df_train)} rows | Test: {len(df_test)} rows")
    print("\nPer-class counts, train vs. test (smallest classes worth watching):")
    counts = pd.DataFrame({
        "train": df_train["rating"].value_counts(),
        "test": df_test["rating"].value_counts(),
    }).fillna(0).astype(int).sort_values("train", ascending=False)
    print(counts)
    smallest = counts.index[-1]
    print(f"\nNote: '{smallest}' has only {counts.loc[smallest, 'test']} test examples -- "
          f"any metric on this class alone will be noisy, not a reliable read on real performance.")

    # --- genre vocabulary from TRAINING data only ---
    train_genre_counts = df_train.explode("genre_list")["genre_list"].value_counts()
    top_genres = train_genre_counts.head(TOP_N_GENRES).index.tolist()
    print(f"\nTop {TOP_N_GENRES} genres (from training data only): {top_genres}")

    genre_train = encode_genres(df_train["genre_list"], top_genres)
    genre_test = encode_genres(df_test["genre_list"], top_genres)

    # --- duration features ---
    duration_train = build_duration_features(df_train).reset_index(drop=True)
    duration_test = build_duration_features(df_test).reset_index(drop=True)

    # --- release_year + duration: scale, fit on TRAIN only ---
    numeric_train = pd.concat([
        df_train[["release_year"]].reset_index(drop=True), duration_train
    ], axis=1)
    numeric_test = pd.concat([
        df_test[["release_year"]].reset_index(drop=True), duration_test
    ], axis=1)

    scaler = StandardScaler()
    numeric_train_scaled = pd.DataFrame(
        scaler.fit_transform(numeric_train), columns=numeric_train.columns
    )
    numeric_test_scaled = pd.DataFrame(
        scaler.transform(numeric_test), columns=numeric_test.columns
    )

    # --- combine ---
    X_train = pd.concat([numeric_train_scaled, genre_train.reset_index(drop=True)], axis=1)
    X_test = pd.concat([numeric_test_scaled, genre_test.reset_index(drop=True)], axis=1)

    print(f"\nFinal feature count: {X_train.shape[1]}")
    print(f"Features: {list(X_train.columns)}")

    # country, director, cast intentionally excluded -- see problem_statement.md
    # (country: dominated by one category + a large "Unknown" bucket, low
    # expected signal for a baseline; director/cast: excluded since Stage 01)

    # --- save ---
    np.save(OUTPUT_DIR / "X_train.npy", X_train.values)
    np.save(OUTPUT_DIR / "X_test.npy", X_test.values)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    joblib.dump({"scaler": scaler, "top_genres": top_genres}, OUTPUT_DIR / "feature_pipeline.joblib")
    with open(OUTPUT_DIR / "feature_names.txt", "w") as f:
        f.write("\n".join(X_train.columns))

    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
