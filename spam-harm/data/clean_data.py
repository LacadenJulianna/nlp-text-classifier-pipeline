"""
spam-harm/data/clean_data.py

Cleans the raw spam/harm dataset and produces the one canonical
train/test split every later phase (baseline, iterate) loads directly
-- splitting happens exactly once, here, so every phase is compared on
the identical held-out set.

Decisions made here:
  1. Drop exact-duplicate messages -- the raw file has 323 of its 1,000
     rows as exact duplicates, all on the spam side (confirmed during
     EDA). Keeping them would let the same message appear in both train
     and test, leaking information, and would also overstate how
     balanced the classes really are: the raw file looks like an even
     500/500 split, but after dedup it's ~500 ham / ~177 spam (about
     74%/26%).
  2. Map label strings to 0/1 (1=spam) -- sklearn's binary metrics
     (f1_score, precision, recall) default to pos_label=1, so spam=1
     makes "the spam-class F1" the default reading of those functions,
     not something requiring an explicit pos_label everywhere.
  3. Strip leading/trailing whitespace only -- no stopword removal or
     stemming here. TfidfVectorizer's own `stop_words` parameter is
     tuned as a grid-search variable in the iterate phase instead of
     being hard-coded here, so its effect can actually be measured
     rather than assumed.
  4. Split is stratified (spam is ~26% of the data post-dedup) and uses
     a fixed random_state so it is exactly reproducible across scripts
     and runs.

Run:
    python clean_data.py
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from load_raw import load_raw_data, RAW_PATH

SCRIPT_DIR = Path(__file__).resolve().parent
CLEAN_DIR = SCRIPT_DIR / "clean"
TRAIN_PATH = CLEAN_DIR / "train.csv"
TEST_PATH = CLEAN_DIR / "test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start_rows = len(df)

    df = df.dropna(subset=["message"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"] != ""]
    df = df.drop_duplicates(subset=["message"])
    df["label"] = df["label"].str.lower().map({"spam": 1, "ham": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    dropped = start_rows - len(df)
    print(f"Rows: {start_rows} -> {len(df)} ({dropped} dropped: empty/duplicate/unmapped label)")
    return df[["label", "message"]].reset_index(drop=True)


def main():
    CLEAN_DIR.mkdir(exist_ok=True)
    raw = load_raw_data(RAW_PATH)
    clean = clean_data(raw)

    print(f"\nClass balance:\n{clean['label'].value_counts(normalize=True)}")

    train_df, test_df = train_test_split(
        clean, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=clean["label"]
    )
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"\nTrain: {len(train_df)} rows -> {TRAIN_PATH}")
    print(f"Test:  {len(test_df)} rows -> {TEST_PATH}")


if __name__ == "__main__":
    main()
