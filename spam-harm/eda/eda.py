"""
spam-harm/eda/eda.py

Explores the cleaned training split: class balance, message-length
distribution by class, and the most distinctive words per class (a
simple frequency count, not TF-IDF weighting -- this is exploratory,
the actual model's TF-IDF weighting happens in baseline/train_baseline.py).

Run:
    python eda.py
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR.parent / "data" / "clean" / "train.csv"
FINDINGS_PATH = SCRIPT_DIR / "findings.md"
CHART_PATH = SCRIPT_DIR / "class_balance.png"

# Small, explicit stopword list for this exploratory word-frequency
# count only -- NOT used by the model itself (TfidfVectorizer's
# stop_words parameter is a separate, tunable choice in the iterate
# phase).
STOPWORDS = {
    "the", "a", "an", "to", "you", "i", "and", "is", "in", "it", "of",
    "for", "on", "your", "my", "me", "we", "at", "be", "this", "that",
    "are", "was", "have", "will", "with", "not", "so", "but", "if",
}


def class_balance(df: pd.DataFrame) -> pd.Series:
    return df["label"].value_counts(normalize=True).sort_index()


def message_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    lengths = df["message"].str.len()
    return df.assign(length=lengths).groupby("label")["length"].describe()


def top_tokens(df: pd.DataFrame, label_value: int, n: int = 15) -> list[tuple[str, int]]:
    subset = df[df["label"] == label_value]["message"]
    counts = Counter()
    for msg in subset:
        words = re.findall(r"[a-z']+", msg.lower())
        counts.update(w for w in words if w not in STOPWORDS)
    return counts.most_common(n)


def main():
    df = pd.read_csv(TRAIN_PATH)

    balance = class_balance(df)
    length_stats = message_length_stats(df)
    ham_tokens = top_tokens(df, label_value=0, n=15)
    spam_tokens = top_tokens(df, label_value=1, n=15)

    print("Class balance:")
    print(balance)
    print("\nMessage length stats by class:")
    print(length_stats)
    print(f"\nTop ham tokens: {ham_tokens}")
    print(f"\nTop spam tokens: {spam_tokens}")

    plt.figure(figsize=(5, 4))
    balance.plot(kind="bar")
    plt.title("Class balance (train set)")
    plt.xlabel("label (0=ham, 1=spam)")
    plt.ylabel("proportion")
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=120)
    print(f"\nSaved chart -> {CHART_PATH}")

    with open(FINDINGS_PATH, "w") as f:
        f.write("# EDA Findings\n\n")
        f.write("## Class balance\n\n")
        f.write(f"```\n{balance}\n```\n\n")
        f.write("![Class balance](class_balance.png)\n\n")
        f.write("## Message length by class\n\n")
        f.write(f"```\n{length_stats}\n```\n\n")
        f.write("## Most distinctive tokens\n\n")
        f.write(f"- **Ham:** {ham_tokens}\n")
        f.write(f"- **Spam:** {spam_tokens}\n")
    print(f"Saved findings -> {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
