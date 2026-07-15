"""
spam-harm/data/load_raw.py

Loads the raw spam/harm dataset (`message_content`/`is_spam`, UTF-8,
`is_spam` already 0/1 int). Also handles the classic SMS Spam
Collection format (`v1`/`v2` or `label`/`message` columns, latin-1) as
a fallback, in case that dataset is swapped in later. Regardless of
source format, always returns `label` as the string "spam"/"ham" (not
0/1) -- clean_data.py owns the one place that maps to int, so every
downstream script sees the same convention no matter which raw format
was read.

Run:
    python load_raw.py
"""

import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / "raw" / "spam.csv"


def load_raw_data(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    if "message_content" in df.columns and "is_spam" in df.columns:
        df = df[["message_content", "is_spam"]].rename(
            columns={"message_content": "message", "is_spam": "label"}
        )
        df["label"] = df["label"].map({1: "spam", 0: "ham"})
    elif "v1" in df.columns and "v2" in df.columns:
        df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    elif "label" in df.columns and "message" in df.columns:
        df = df[["label", "message"]]
    elif df.shape[1] == 2:
        df.columns = ["label", "message"]
    else:
        raise ValueError(
            f"Unrecognized dataset format at {path}: columns are "
            f"{list(df.columns)}. Expected message_content/is_spam, "
            "Kaggle's v1/v2 columns, or a plain label/message CSV."
        )
    return df[["label", "message"]]


def main():
    df = load_raw_data(RAW_PATH)
    print(f"Loaded {len(df)} rows from {RAW_PATH}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
