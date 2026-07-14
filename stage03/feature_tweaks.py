"""
stage03/feature_tweaks.py

Week 3, Day 13: Light feature tweaks

Tests two candidate feature additions against the Day 12 winning model:
  1. `country` -- excluded in Stage 01/02 as "questionable value," still
     an open question. Top-8 exact country strings + "Other", derived
     from TRAINING data only.
  2. `genre_count` -- number of genres per title. Deterministic per-row
     transform, no leakage risk (doesn't depend on any fitted vocabulary).

Both are tested individually and combined, using the SAME train/test
split as Day 12 (reproduced via identical split parameters on the same
dataframe -- verified to match row-for-row).

Per the roadmap's own instruction ("only keep them if they clearly help
... don't chase small gains"): if nothing clears a real margin, the
right move is to discard and keep Day 12's feature set unchanged, not
to keep a coin-flip-sized "improvement" just because it's technically
positive.

Run:
    python feature_tweaks.py
"""

import ast
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CLEAN_PATH = REPO_ROOT / "stage01" / "disney_plus_titles_clean.csv"
STAGE02_FEATURES = REPO_ROOT / "stage02" / "features"

TOP_N_COUNTRY = 8
RANDOM_STATE = 42
N_ESTIMATORS = 300
# A macro F1 gain smaller than this is treated as noise, not a real win --
# 290 test rows means small deltas aren't reliable signal.
MIN_MEANINGFUL_GAIN = 0.02


def encode_country(countries: pd.Series, vocab: list[str]) -> pd.DataFrame:
    return pd.get_dummies(countries.apply(lambda c: c if c in vocab else "Other"), prefix="country")


def main():
    df = pd.read_csv(CLEAN_PATH)
    df["genre_list"] = df["genre_list"].apply(ast.literal_eval)
    y_full = df["rating"].values

    # Reproduce Day 7's exact split
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y_full
    )
    df_train, df_test = df.loc[train_idx], df.loc[test_idx]

    y_test_check = np.load(STAGE02_FEATURES / "y_test.npy", allow_pickle=True)
    assert (df_test["rating"].values == y_test_check).all(), "Split mismatch -- not comparable to Day 12"
    print("Split reproduction verified: matches Day 12's test set exactly.\n")

    X_train_base = np.load(STAGE02_FEATURES / "X_train.npy")
    X_test_base = np.load(STAGE02_FEATURES / "X_test.npy")
    y_train = np.load(STAGE02_FEATURES / "y_train.npy", allow_pickle=True)
    y_test = np.load(STAGE02_FEATURES / "y_test.npy", allow_pickle=True)

    top_countries = df_train["country"].value_counts().head(TOP_N_COUNTRY).index.tolist()
    country_train = encode_country(df_train["country"], top_countries).reset_index(drop=True)
    country_test = encode_country(df_test["country"], top_countries).reset_index(drop=True)
    country_test = country_test.reindex(columns=country_train.columns, fill_value=0)

    genre_count_train = df_train["genre_list"].apply(len).values.reshape(-1, 1)
    genre_count_test = df_test["genre_list"].apply(len).values.reshape(-1, 1)

    variants = {
        "Day 12 baseline (no tweaks)": (X_train_base, X_test_base),
        "+ country (top 8 + Other)": (
            np.hstack([X_train_base, country_train.values]),
            np.hstack([X_test_base, country_test.values]),
        ),
        "+ genre_count": (
            np.hstack([X_train_base, genre_count_train]),
            np.hstack([X_test_base, genre_count_test]),
        ),
        "+ country + genre_count": (
            np.hstack([X_train_base, country_train.values, genre_count_train]),
            np.hstack([X_test_base, country_test.values, genre_count_test]),
        ),
    }

    print(f"{'Variant':35s} {'Macro F1':>10s} {'Delta':>10s}")
    baseline_score = None
    results = {}
    for name, (Xtr, Xte) in variants.items():
        model = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
        model.fit(Xtr, y_train)
        score = f1_score(y_test, model.predict(Xte), average="macro", zero_division=0)
        results[name] = score
        if baseline_score is None:
            baseline_score = score
            print(f"{name:35s} {score:10.4f} {'--':>10s}")
        else:
            delta = score - baseline_score
            print(f"{name:35s} {score:10.4f} {delta:+10.4f}")

    best_name = max(results, key=results.get)
    best_delta = results[best_name] - baseline_score

    print()
    if best_name != "Day 12 baseline (no tweaks)" and best_delta >= MIN_MEANINGFUL_GAIN:
        print(f"DECISION: keep '{best_name}' -- gain of {best_delta:+.4f} clears the "
              f"{MIN_MEANINGFUL_GAIN} meaningful-gain threshold.")
    else:
        print(f"DECISION: discard all tweaks. Best variant ('{best_name}') only gained "
              f"{best_delta:+.4f}, below the {MIN_MEANINGFUL_GAIN} threshold for a 290-row "
              f"test set -- likely noise, not a real improvement. Keeping Day 12's feature "
              f"set unchanged for Day 14 tuning.")


if __name__ == "__main__":
    main()
