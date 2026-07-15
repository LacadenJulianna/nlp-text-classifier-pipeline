# Spam/Harm Text Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a spam/harm SMS text classifier (TF-IDF + classical ML) through the full arc from raw data to a live, verified Streamlit deployment, in a new `spam-harm/` folder.

**Architecture:** A linear data pipeline (raw CSV → cleaned/split CSVs → baseline model → tuned candidate model → final model) followed by a thin inference wrapper (`predict.py`) consumed by a Streamlit app. Baseline and candidate models are both full `sklearn.pipeline.Pipeline` objects bundling `TfidfVectorizer` + classifier as one artifact, so inference never needs to separately manage a fitted vectorizer.

**Tech Stack:** Python, pandas, scikit-learn (`TfidfVectorizer`, `MultinomialNB`, `LogisticRegression`, `GridSearchCV`), joblib, matplotlib/seaborn, Streamlit.

## Global Constraints

- `RANDOM_STATE = 42` and `TEST_SIZE = 0.2` everywhere a split or a seeded model is used (matches `stage02`/`stage03` convention).
- Headline metric is **F1 on the spam class** (`pos_label=1`), never raw accuracy — the raw file is an even 500/500 split, but 323 of the 500 "spam" rows are exact duplicates; once `clean_data.py` dedupes, the real class balance is ~74% ham / ~26% spam, imbalanced enough that accuracy alone would be misleading.
- No pytest. Tests are plain assert-based scripts with a `run_checks()`/`run()` function, a list of `(name, passed)` tuples, PASS/FAIL printout, and `raise SystemExit` on any failure — matching `stage01/test_clean_data.py` and `stage04/stress_test.py`.
- Every model artifact is a full `sklearn.pipeline.Pipeline` (vectorizer + classifier bundled) saved via `joblib.dump`. Inference only ever calls `.predict()`/`.predict_proba()` on raw text — never re-fits anything.
- Classical ML only (TF-IDF + Naive Bayes / Logistic Regression) — no embeddings, no transformers.
- The raw dataset is sourced by the human, not fetched by an agent (no internet access assumed in the execution environment).
- Use `Path(__file__).resolve().parent` (`SCRIPT_DIR`) for all path handling, matching every existing script in this repo.
- Always pass `encoding="utf-8"` to `open()`/`.read_text()`/`.write_text()` calls on generated markdown files. Windows defaults to cp1252, which silently corrupts em dashes (`—`) into mojibake on write and raises `UnicodeDecodeError` on read on non-Windows machines — found and fixed in Task 5's `evaluate.py`/`test_evaluate.py`.
- Every `joblib.load()` in this plan loads a model artifact this same pipeline trained and committed (`baseline_model.joblib`, `candidate_model.joblib`, `final_model.joblib`) — never a file from an untrusted external source. This matches the existing pattern in `stage04/predict.py`. Do not load `.joblib`/pickle files from any source outside this repo's own training scripts.

---

### Task 1: Raw dataset loader

**Dataset in use:** `spam-harm/data/raw/spam.csv` (1,000 rows, columns `message_content` + `is_spam` as 0/1 int, UTF-8 encoded, longer email-style text rather than short SMS, 323 exact-duplicate rows — all on the spam side). The loader below is written for this specific schema, with a fallback path for the classic SMS Spam Collection format (`v1`/`v2` or `label`/`message` columns, latin-1) in case that dataset is used instead in the future.

**Files:**
- Create: `spam-harm/data/raw/.gitkeep` (placeholder so the folder exists in git even before the CSV is added)
- Create: `spam-harm/data/load_raw.py`
- Test: `spam-harm/data/test_load_raw.py`

**Interfaces:**
- Produces: `load_raw_data(path: Path) -> pd.DataFrame` — returns a DataFrame with exactly two columns, `label` (raw string, "spam"/"ham") and `message` (raw string, uncleaned). Raises `ValueError` if the format isn't recognized.

- [ ] **Step 1: Create the raw data folder placeholder**

```bash
mkdir -p spam-harm/data/raw
touch spam-harm/data/raw/.gitkeep
```

- [ ] **Step 2: Write the failing test**

Create `spam-harm/data/test_load_raw.py`:

```python
"""
spam-harm/data/test_load_raw.py

Sanity checks for load_raw.py. Run after any change to confirm the loader
still produces the expected two-column shape, regardless of which raw
distribution (Kaggle CSV or plain label/message CSV) is on disk.

Run:
    python test_load_raw.py
"""

import pandas as pd
from pathlib import Path
from load_raw import load_raw_data

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_PATH = SCRIPT_DIR / "raw" / "spam.csv"


def run_checks():
    if not RAW_PATH.exists():
        raise SystemExit(
            f"Raw dataset not found at {RAW_PATH}. Place the raw dataset CSV "
            "there before running this test (see Task 1's dataset note in "
            "the implementation plan)."
        )

    df = load_raw_data(RAW_PATH)

    checks = [
        ("has exactly two columns", list(df.columns) == ["label", "message"]),
        ("has at least 900 rows", len(df) >= 900),
        ("label column only has spam/ham values",
         set(df["label"].str.lower().unique()) <= {"spam", "ham"}),
        ("message column has no nulls", df["message"].isnull().sum() == 0),
        ("both spam and ham present", set(df["label"].str.lower().unique()) == {"spam", "ham"}),
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd spam-harm/data && python test_load_raw.py`
Expected: `ModuleNotFoundError: No module named 'load_raw'` (the module doesn't exist yet), or if the raw file is also missing, the `SystemExit` about the missing dataset fires first once `load_raw.py` exists as an empty stub. Confirm it fails for the right reason (missing module) before writing the implementation.

- [ ] **Step 4: Write minimal implementation**

Create `spam-harm/data/load_raw.py`:

```python
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd spam-harm/data && python test_load_raw.py`
Expected: `5/5 passed.` (assuming the raw file is in place per the prerequisite)

- [ ] **Step 6: Commit**

```bash
git add spam-harm/data/raw/.gitkeep spam-harm/data/load_raw.py spam-harm/data/test_load_raw.py
git commit -m "feat(spam-harm): add raw spam/harm dataset loader"
```

Note: `spam-harm/data/raw/spam.csv` itself should also be committed once present (small file, matches how `disney_plus_titles.csv` is committed at repo root) — add it in Task 2's commit alongside the cleaning script, once its presence has been confirmed by this task's test.

---

### Task 2: Cleaning + train/test split

**Files:**
- Create: `spam-harm/data/clean_data.py`
- Create: `spam-harm/data/test_clean_data.py`

**Interfaces:**
- Consumes: `load_raw_data(path) -> pd.DataFrame` from Task 1 (`spam-harm/data/load_raw.py`)
- Produces: `clean_data(df: pd.DataFrame) -> pd.DataFrame` — returns a DataFrame with columns `label` (int, 1=spam/0=ham) and `message` (str, cleaned). Also produces `spam-harm/data/clean/train.csv` and `spam-harm/data/clean/test.csv` (columns `label`,`message`) via `main()`, which every later phase (baseline, iterate) loads directly — this is the one canonical split, never re-split elsewhere.

- [ ] **Step 1: Write the failing test**

Create `spam-harm/data/test_clean_data.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spam-harm/data && python test_clean_data.py`
Expected: `ModuleNotFoundError: No module named 'clean_data'`

- [ ] **Step 3: Write minimal implementation**

Create `spam-harm/data/clean_data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spam-harm/data && python test_clean_data.py`
Expected: `6/6 passed.`

- [ ] **Step 5: Generate the canonical split**

Run: `cd spam-harm/data && python clean_data.py`
Expected: prints row-drop summary, class balance, and confirms `spam-harm/data/clean/train.csv` and `test.csv` were written.

- [ ] **Step 6: Commit**

```bash
git add -f spam-harm/data/raw/spam.csv
git add spam-harm/data/clean_data.py spam-harm/data/test_clean_data.py spam-harm/data/clean/train.csv spam-harm/data/clean/test.csv
git commit -m "feat(spam-harm): add cleaning pipeline and canonical train/test split"
```

---

### Task 3: EDA

**Files:**
- Create: `spam-harm/eda/eda.py`
- Create: `spam-harm/eda/test_eda.py`

**Interfaces:**
- Consumes: `spam-harm/data/clean/train.csv` (columns `label`,`message`) from Task 2
- Produces: `class_balance(df) -> pd.Series`, `message_length_stats(df) -> pd.DataFrame`, `top_tokens(df, label_value, n=15) -> list[tuple[str, int]]` — pure functions, reused by nothing downstream (EDA is terminal), but kept as functions so they're independently testable on a small fixture instead of only being verifiable by eyeballing printed output.

- [ ] **Step 1: Write the failing test**

Create `spam-harm/eda/test_eda.py`:

```python
"""
spam-harm/eda/test_eda.py

Unit tests for eda.py's pure functions, using a small inline fixture
instead of the full dataset -- these are fast, isolated checks of the
computation logic, independent of what's actually in the real data.

Run:
    python test_eda.py
"""

import pandas as pd
from eda import class_balance, message_length_stats, top_tokens

FIXTURE = pd.DataFrame({
    "label": [0, 0, 0, 1, 1],
    "message": [
        "hey are we still on for lunch",
        "call me when you get a chance",
        "see you tomorrow",
        "WINNER call now to claim your free prize",
        "free entry to win a prize call now",
    ],
})


def run_checks():
    balance = class_balance(FIXTURE)
    length_stats = message_length_stats(FIXTURE)
    spam_tokens = top_tokens(FIXTURE, label_value=1, n=3)

    checks = [
        ("class_balance sums to 1.0", abs(balance.sum() - 1.0) < 1e-9),
        ("class_balance has both classes", set(balance.index) == {0, 1}),
        ("length_stats has one row per class", len(length_stats) == 2),
        ("length_stats has a mean column", "mean" in length_stats.columns),
        ("top_tokens returns requested count or fewer", len(spam_tokens) <= 3),
        ("top_tokens finds 'call' among top spam tokens",
         any(tok == "call" for tok, _ in spam_tokens)),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spam-harm/eda && python test_eda.py`
Expected: `ModuleNotFoundError: No module named 'eda'`

- [ ] **Step 3: Write minimal implementation**

Create `spam-harm/eda/eda.py`:

```python
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

    with open(FINDINGS_PATH, "w", encoding="utf-8") as f:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spam-harm/eda && python test_eda.py`
Expected: `6/6 passed.`

- [ ] **Step 5: Generate real findings**

Run: `cd spam-harm/eda && python eda.py`
Expected: prints class balance, length stats, top tokens; writes `findings.md` and `class_balance.png`.

- [ ] **Step 6: Commit**

```bash
git add spam-harm/eda/eda.py spam-harm/eda/test_eda.py spam-harm/eda/findings.md spam-harm/eda/class_balance.png
git commit -m "feat(spam-harm): add EDA on class balance, message length, and token frequency"
```

---

### Task 4: Baseline model (TF-IDF + Naive Bayes)

**Files:**
- Create: `spam-harm/baseline/train_baseline.py`
- Create: `spam-harm/baseline/test_train_baseline.py`

**Interfaces:**
- Consumes: `spam-harm/data/clean/train.csv` from Task 2
- Produces: `spam-harm/baseline/baseline_model.joblib` — a fitted `sklearn.pipeline.Pipeline` with steps `("tfidf", TfidfVectorizer())` and `("clf", MultinomialNB())`. Exposes `.predict(list[str])` and `.predict_proba(list[str])` directly on raw text. Consumed by Task 5 (evaluate) and Task 7 (model comparison).

- [ ] **Step 1: Write the failing test**

Create `spam-harm/baseline/test_train_baseline.py`:

```python
"""
spam-harm/baseline/test_train_baseline.py

Confirms the baseline pipeline trains and produces a usable model --
not a real evaluation (see evaluate.py for that), just a sanity check
that the artifact is a working Pipeline.

Run:
    python test_train_baseline.py
"""

import subprocess
import sys
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "train_baseline.py")], check=True)

    model = joblib.load(MODEL_PATH)
    pred = model.predict(["free entry to win a prize call now"])
    proba = model.predict_proba(["hey are we still on for lunch"])

    checks = [
        ("model file was created", MODEL_PATH.exists()),
        ("model has predict()", hasattr(model, "predict")),
        ("model has predict_proba()", hasattr(model, "predict_proba")),
        ("predict returns one label per input", len(pred) == 1),
        ("predict_proba returns two class probabilities", proba.shape[1] == 2),
        ("probabilities sum to ~1.0", abs(proba[0].sum() - 1.0) < 1e-6),
    ]

    print(f"\nRunning {len(checks)} checks...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spam-harm/baseline && python test_train_baseline.py`
Expected: fails because `train_baseline.py` doesn't exist yet (`FileNotFoundError` from `subprocess.run`).

- [ ] **Step 3: Write minimal implementation**

Create `spam-harm/baseline/train_baseline.py`:

```python
"""
spam-harm/baseline/train_baseline.py

Trains the simplest reasonable model for spam/ham text classification:
TF-IDF features + Multinomial Naive Bayes, the classic pairing for this
exact problem (fast, works well on short text, a well-understood
default before reaching for anything tuned).

The vectorizer and classifier are bundled into ONE sklearn Pipeline and
saved as a single artifact -- inference (predict.py) never needs to
separately manage a fitted vectorizer, it just calls .predict() on raw
text through the pipeline.

This script trains and saves the model only. Full evaluation is a
separate step -- see evaluate.py.

Run:
    python train_baseline.py
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR.parent / "data" / "clean" / "train.csv"
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"

RANDOM_STATE = 42


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Training on {len(train_df)} rows")

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB()),
    ])
    model.fit(train_df["message"], train_df["label"])

    # Rough sanity check only -- NOT the real evaluation, which needs
    # the held-out test set (evaluate.py).
    train_pred = model.predict(train_df["message"])
    train_f1 = f1_score(train_df["label"], train_pred, pos_label=1)
    print(f"Training-set spam F1 (sanity check only, not a real metric): {train_f1:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spam-harm/baseline && python test_train_baseline.py`
Expected: `6/6 passed.`

- [ ] **Step 5: Commit**

```bash
git add spam-harm/baseline/train_baseline.py spam-harm/baseline/test_train_baseline.py spam-harm/baseline/baseline_model.joblib
git commit -m "feat(spam-harm): add TF-IDF + Naive Bayes baseline model"
```

---

### Task 5: Baseline evaluation

**Files:**
- Create: `spam-harm/baseline/evaluate.py`
- Create: `spam-harm/baseline/test_evaluate.py`

**Interfaces:**
- Consumes: `spam-harm/data/clean/test.csv` (Task 2), `spam-harm/baseline/baseline_model.joblib` (Task 4)
- Produces: `spam-harm/baseline/metrics_report.md`, `spam-harm/baseline/confusion_matrix.png`. The spam-F1 number recorded here is what Task 7 compares the candidate model against.

- [ ] **Step 1: Write the failing test**

Create `spam-harm/baseline/test_evaluate.py`:

```python
"""
spam-harm/baseline/test_evaluate.py

Confirms evaluate.py produces the expected output files with the
expected structure.

Run:
    python test_evaluate.py
"""

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR / "metrics_report.md"
CM_PATH = SCRIPT_DIR / "confusion_matrix.png"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "evaluate.py")], check=True)

    report_text = REPORT_PATH.read_text(encoding="utf-8")

    checks = [
        ("metrics report was created", REPORT_PATH.exists()),
        ("confusion matrix image was created", CM_PATH.exists()),
        ("report mentions spam F1", "Spam F1" in report_text or "spam F1" in report_text.lower()),
        ("report mentions naive baseline comparison", "naive" in report_text.lower()),
    ]

    print(f"\nRunning {len(checks)} checks...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spam-harm/baseline && python test_evaluate.py`
Expected: fails with `FileNotFoundError` (`evaluate.py` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `spam-harm/baseline/evaluate.py`:

```python
"""
spam-harm/baseline/evaluate.py

Evaluates the baseline model against the held-out test set. Headline
metric is spam-class F1, not accuracy -- a model that always predicts
"ham" would score accuracy equal to the ham proportion of the test set
while catching zero spam, which is exactly why accuracy alone is the
wrong number to lead with here. The naive baseline below makes that
concrete: it's the "always predict ham" F1, which is 0.0 by
construction (zero recall on the spam class). The ham proportion is
computed from the actual test set below rather than hard-coded, since
it depends on which raw dataset is in use.

Run:
    python evaluate.py

Produces:
    spam-harm/baseline/metrics_report.md
    spam-harm/baseline/confusion_matrix.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_PATH = SCRIPT_DIR.parent / "data" / "clean" / "test.csv"
MODEL_PATH = SCRIPT_DIR / "baseline_model.joblib"
REPORT_PATH = SCRIPT_DIR / "metrics_report.md"
CM_PATH = SCRIPT_DIR / "confusion_matrix.png"

# "Always predict ham" -- zero recall on spam, so spam F1 = 0.0 by
# construction. The number to beat.
NAIVE_BASELINE_SPAM_F1 = 0.0


def main():
    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    y_test = test_df["label"]
    y_pred = model.predict(test_df["message"])
    ham_proportion = (y_test == 0).mean()

    acc = accuracy_score(y_test, y_pred)
    spam_f1 = f1_score(y_test, y_pred, pos_label=1)
    spam_precision = precision_score(y_test, y_pred, pos_label=1)
    spam_recall = recall_score(y_test, y_pred, pos_label=1)
    report = classification_report(y_test, y_pred, target_names=["ham", "spam"])

    print("=" * 60)
    print("BASELINE MODEL — TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy:       {acc:.4f} ({acc*100:.1f}%)")
    print(f"Spam F1:        {spam_f1:.4f}")
    print(f"Spam precision: {spam_precision:.4f}  (of messages flagged spam, how many really are)")
    print(f"Spam recall:    {spam_recall:.4f}  (of real spam, how much got caught)")
    print(f"\nNaive baseline (always predict ham): spam F1 = {NAIVE_BASELINE_SPAM_F1:.4f}")
    print()
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Baseline model — confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(CM_PATH, dpi=120)
    print(f"\nSaved confusion matrix -> {CM_PATH}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Baseline Model — Metrics Report\n\n")
        f.write(f"- **Accuracy:** {acc:.4f} ({acc*100:.1f}%)\n")
        f.write(f"- **Spam F1:** {spam_f1:.4f}\n")
        f.write(f"- **Spam precision:** {spam_precision:.4f}\n")
        f.write(f"- **Spam recall:** {spam_recall:.4f}\n\n")
        f.write(f"**Naive baseline** (always predict ham): spam F1 = {NAIVE_BASELINE_SPAM_F1:.4f}\n\n")
        f.write(f"Accuracy alone is misleading here: this test set is {ham_proportion*100:.1f}% ham, so a "
                f"model that always predicts \"ham\" would score {ham_proportion*100:.1f}% accuracy while "
                "catching zero spam. Spam F1 is the number that actually reflects "
                "whether the model is useful.\n\n")
        f.write("## Per-class breakdown\n\n```\n")
        f.write(report)
        f.write("\n```\n\n")
        f.write("![Confusion matrix](confusion_matrix.png)\n")
    print(f"Saved metrics report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spam-harm/baseline && python test_evaluate.py`
Expected: `4/4 passed.`

- [ ] **Step 5: Commit**

```bash
git add spam-harm/baseline/evaluate.py spam-harm/baseline/test_evaluate.py spam-harm/baseline/metrics_report.md spam-harm/baseline/confusion_matrix.png
git commit -m "feat(spam-harm): add baseline evaluation with spam-F1 headline metric"
```

---

### Task 6: Iteration — tuned Logistic Regression candidate

**Files:**
- Create: `spam-harm/iterate/train_candidate.py`
- Create: `spam-harm/iterate/test_train_candidate.py`

**Interfaces:**
- Consumes: `spam-harm/data/clean/train.csv` from Task 2
- Produces: `spam-harm/iterate/candidate_model.joblib` (fitted `Pipeline` with `TfidfVectorizer` + `LogisticRegression`, same `.predict()`/`.predict_proba()` interface as the baseline), `spam-harm/iterate/tuning_log.md`. Consumed by Task 7.

**Design note folded into this task (per spec's request to test whether stopword removal helps):** rather than a separate script, `tfidf__stop_words` is one axis of the same grid search alongside `tfidf__ngram_range` and `clf__C` — `grid.best_params_` and `grid.cv_results_` directly answer "did removing stopwords help" as a natural byproduct of tuning, instead of duplicating the training code in a second script. Stemming (also mentioned in the spec) is deliberately **not** implemented: it would require an extra dependency (`nltk`, plus a corpus download) for a "NLP-lite" project, so this tradeoff is documented in `tuning_log.md` rather than silently dropped.

- [ ] **Step 1: Write the failing test**

Create `spam-harm/iterate/test_train_candidate.py`:

```python
"""
spam-harm/iterate/test_train_candidate.py

Confirms the tuned candidate pipeline trains and produces a usable
model, and that the tuning log actually documents the stopword-removal
comparison (not just the winning params).

Run:
    python test_train_candidate.py
"""

import subprocess
import sys
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "candidate_model.joblib"
LOG_PATH = SCRIPT_DIR / "tuning_log.md"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "train_candidate.py")], check=True)

    model = joblib.load(MODEL_PATH)
    pred = model.predict(["free entry to win a prize call now"])
    log_text = LOG_PATH.read_text(encoding="utf-8")

    checks = [
        ("model file was created", MODEL_PATH.exists()),
        ("model has predict()", hasattr(model, "predict")),
        ("predict returns one label per input", len(pred) == 1),
        ("tuning log was created", LOG_PATH.exists()),
        ("tuning log documents stopword comparison", "stop_words" in log_text.lower() or "stopword" in log_text.lower()),
        ("tuning log documents stemming decision", "stemming" in log_text.lower()),
    ]

    print(f"\nRunning {len(checks)} checks...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spam-harm/iterate && python test_train_candidate.py`
Expected: fails with `FileNotFoundError` (`train_candidate.py` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `spam-harm/iterate/train_candidate.py`:

```python
"""
spam-harm/iterate/train_candidate.py

Trains a tuned Logistic Regression candidate on the same TF-IDF
features as the baseline, via grid search over:
  - clf__C: regularization strength
  - tfidf__ngram_range: unigrams only vs. unigrams+bigrams
  - tfidf__stop_words: None vs. "english" -- this is how the "does
    stopword removal help" question gets answered, as a natural
    byproduct of the same search rather than a separate experiment.

Stemming was considered (per the design spec) but not implemented: it
needs an extra dependency (nltk + a corpus download) that doesn't pay
for itself on a "NLP-lite" project this size. Documented here, not
silently dropped.

Run:
    python train_candidate.py
"""

import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = SCRIPT_DIR.parent / "data" / "clean" / "train.csv"
MODEL_PATH = SCRIPT_DIR / "candidate_model.joblib"
LOG_PATH = SCRIPT_DIR / "tuning_log.md"

RANDOM_STATE = 42

PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__stop_words": [None, "english"],
    "clf__C": [0.01, 0.1, 1, 10],
}


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    print(f"Training on {len(train_df)} rows")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, PARAM_GRID, scoring="f1", cv=cv, n_jobs=-1)
    grid.fit(train_df["message"], train_df["label"])

    print(f"Best CV spam F1: {grid.best_score_:.4f}")
    print(f"Best params: {grid.best_params_}")

    # Isolate the stopword-removal effect specifically: mean CV score
    # for stop_words=None vs. "english", averaged across every other
    # param combination, so the comparison isn't just "the single best
    # row happened to use X."
    results = pd.DataFrame(grid.cv_results_)
    stopword_effect = results.groupby(
        results["params"].apply(lambda p: p["tfidf__stop_words"])
    )["mean_test_score"].mean()
    print(f"\nMean CV spam F1 by stop_words setting:\n{stopword_effect}")

    joblib.dump(grid.best_estimator_, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("# Candidate Model — Tuning Log\n\n")
        f.write("## Grid searched\n\n")
        f.write(f"```\n{PARAM_GRID}\n```\n\n")
        f.write(f"## Best result\n\n")
        f.write(f"- **Best CV spam F1:** {grid.best_score_:.4f}\n")
        f.write(f"- **Best params:** {grid.best_params_}\n\n")
        f.write("## Stopword removal: does it help?\n\n")
        f.write(f"Mean CV spam F1 by `tfidf__stop_words` setting (averaged over all "
                f"other grid params, not just the single best row):\n\n")
        f.write(f"```\n{stopword_effect}\n```\n\n")
        best_stopword_setting = stopword_effect.idxmax()
        f.write(f"`{best_stopword_setting}` performs better on average -- "
                f"{'kept' if best_stopword_setting == grid.best_params_['tfidf__stop_words'] else 'note: differs from the single best combination, which used a different setting; the single-best row can differ from the on-average-better setting when it interacts with other params -- best_params_ is what actually gets used.'}\n\n")
        f.write("## Stemming: considered and discarded\n\n")
        f.write("Not implemented for this candidate. Stemming (e.g. via NLTK's "
                "PorterStemmer) would require an extra dependency plus a corpus "
                "download, which doesn't pay for itself on a dataset and project "
                "this size (\"NLP-lite\" per the project's own scope). Worth "
                "revisiting only if a future iteration shows TF-IDF token sparsity "
                "is actually a bottleneck.\n")
    print(f"Saved tuning log -> {LOG_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spam-harm/iterate && python test_train_candidate.py`
Expected: `6/6 passed.` (grid search over 16 combinations × 5 folds = 80 fits; should complete in well under a minute on a dataset this size)

- [ ] **Step 5: Commit**

```bash
git add spam-harm/iterate/train_candidate.py spam-harm/iterate/test_train_candidate.py spam-harm/iterate/candidate_model.joblib spam-harm/iterate/tuning_log.md
git commit -m "feat(spam-harm): add tuned Logistic Regression candidate with stopword-removal comparison"
```

---

### Task 7: Candidate evaluation + final model selection

**Files:**
- Create: `spam-harm/iterate/evaluate_candidate.py`
- Create: `spam-harm/iterate/test_evaluate_candidate.py`

**Interfaces:**
- Consumes: `spam-harm/data/clean/test.csv` (Task 2), `spam-harm/baseline/baseline_model.joblib` (Task 4), `spam-harm/iterate/candidate_model.joblib` (Task 6)
- Produces: `spam-harm/iterate/candidate_metrics.md`, `spam-harm/iterate/model_choice.md`, `spam-harm/iterate/final_model.joblib` (a copy of whichever model — baseline or candidate — wins on test-set spam F1). **Task 8 (`predict.py`) loads `final_model.joblib`, not `candidate_model.joblib` directly** — this is the one artifact everything downstream depends on.

- [ ] **Step 1: Write the failing test**

Create `spam-harm/iterate/test_evaluate_candidate.py`:

```python
"""
spam-harm/iterate/test_evaluate_candidate.py

Confirms the comparison script produces final_model.joblib and
documents the decision honestly (per-class, not just the headline
number).

Run:
    python test_evaluate_candidate.py
"""

import subprocess
import sys
import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FINAL_MODEL_PATH = SCRIPT_DIR / "final_model.joblib"
CHOICE_PATH = SCRIPT_DIR / "model_choice.md"
METRICS_PATH = SCRIPT_DIR / "candidate_metrics.md"


def run_checks():
    subprocess.run([sys.executable, str(SCRIPT_DIR / "evaluate_candidate.py")], check=True)

    final_model = joblib.load(FINAL_MODEL_PATH)
    choice_text = CHOICE_PATH.read_text(encoding="utf-8")

    checks = [
        ("final model file was created", FINAL_MODEL_PATH.exists()),
        ("final model can predict", len(final_model.predict(["free prize call now"])) == 1),
        ("candidate metrics report was created", METRICS_PATH.exists()),
        ("model choice doc was created", CHOICE_PATH.exists()),
        ("model choice doc has a comparison table", "|" in choice_text),
        ("model choice doc states a decision", "chosen" in choice_text.lower() or "selected" in choice_text.lower()),
    ]

    print(f"\nRunning {len(checks)} checks...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spam-harm/iterate && python test_evaluate_candidate.py`
Expected: fails with `FileNotFoundError` (`evaluate_candidate.py` doesn't exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `spam-harm/iterate/evaluate_candidate.py`:

```python
"""
spam-harm/iterate/evaluate_candidate.py

Evaluates the tuned candidate against the same held-out test set the
baseline was scored on, compares both honestly (per-class, not just
the headline spam F1), and copies whichever model wins to
final_model.joblib -- the one artifact predict.py and the Streamlit
app actually load.

Run:
    python evaluate_candidate.py

Produces:
    spam-harm/iterate/candidate_metrics.md
    spam-harm/iterate/model_choice.md
    spam-harm/iterate/final_model.joblib
"""

import shutil
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_PATH = SCRIPT_DIR.parent / "data" / "clean" / "test.csv"
BASELINE_MODEL_PATH = SCRIPT_DIR.parent / "baseline" / "baseline_model.joblib"
CANDIDATE_MODEL_PATH = SCRIPT_DIR / "candidate_model.joblib"
FINAL_MODEL_PATH = SCRIPT_DIR / "final_model.joblib"
METRICS_PATH = SCRIPT_DIR / "candidate_metrics.md"
CHOICE_PATH = SCRIPT_DIR / "model_choice.md"


def score(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "spam_f1": f1_score(y_test, y_pred, pos_label=1),
        "spam_precision": precision_score(y_test, y_pred, pos_label=1),
        "spam_recall": recall_score(y_test, y_pred, pos_label=1),
        "report": classification_report(y_test, y_pred, target_names=["ham", "spam"]),
    }


def main():
    test_df = pd.read_csv(TEST_PATH)
    X_test, y_test = test_df["message"], test_df["label"]

    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    candidate_model = joblib.load(CANDIDATE_MODEL_PATH)

    baseline_scores = score(baseline_model, X_test, y_test)
    candidate_scores = score(candidate_model, X_test, y_test)

    print("Baseline spam F1:  ", f"{baseline_scores['spam_f1']:.4f}")
    print("Candidate spam F1: ", f"{candidate_scores['spam_f1']:.4f}")

    winner = "candidate" if candidate_scores["spam_f1"] >= baseline_scores["spam_f1"] else "baseline"
    winner_path = CANDIDATE_MODEL_PATH if winner == "candidate" else BASELINE_MODEL_PATH
    shutil.copy(winner_path, FINAL_MODEL_PATH)
    print(f"\nWinner: {winner} (copied to {FINAL_MODEL_PATH})")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("# Candidate Model — Metrics Report\n\n")
        f.write(f"- **Accuracy:** {candidate_scores['accuracy']:.4f}\n")
        f.write(f"- **Spam F1:** {candidate_scores['spam_f1']:.4f}\n")
        f.write(f"- **Spam precision:** {candidate_scores['spam_precision']:.4f}\n")
        f.write(f"- **Spam recall:** {candidate_scores['spam_recall']:.4f}\n\n")
        f.write("## Per-class breakdown\n\n```\n")
        f.write(candidate_scores["report"])
        f.write("\n```\n")

    with open(CHOICE_PATH, "w", encoding="utf-8") as f:
        f.write("# Model Choice: Baseline vs. Candidate\n\n")
        f.write("| Metric | Baseline (Naive Bayes) | Candidate (tuned Logistic Regression) |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Accuracy | {baseline_scores['accuracy']:.4f} | {candidate_scores['accuracy']:.4f} |\n")
        f.write(f"| Spam F1 | {baseline_scores['spam_f1']:.4f} | {candidate_scores['spam_f1']:.4f} |\n")
        f.write(f"| Spam precision | {baseline_scores['spam_precision']:.4f} | {candidate_scores['spam_precision']:.4f} |\n")
        f.write(f"| Spam recall | {baseline_scores['spam_recall']:.4f} | {candidate_scores['spam_recall']:.4f} |\n\n")
        f.write(f"**Selected: {winner}**, based on spam F1 on the held-out test set "
                f"(the metric this project treats as the headline number, not accuracy).\n\n")
        f.write("## Per-class breakdown, baseline\n\n```\n")
        f.write(baseline_scores["report"])
        f.write("\n```\n\n## Per-class breakdown, candidate\n\n```\n")
        f.write(candidate_scores["report"])
        f.write("\n```\n")

    print(f"Saved -> {METRICS_PATH}, {CHOICE_PATH}, {FINAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spam-harm/iterate && python test_evaluate_candidate.py`
Expected: `6/6 passed.`

- [ ] **Step 5: Commit**

```bash
git add spam-harm/iterate/evaluate_candidate.py spam-harm/iterate/test_evaluate_candidate.py spam-harm/iterate/candidate_metrics.md spam-harm/iterate/model_choice.md spam-harm/iterate/final_model.joblib
git commit -m "feat(spam-harm): add candidate evaluation and select final model"
```

---

### Task 8: Inference wrapper (`predict.py`)

**Files:**
- Create: `spam-harm/app/predict.py`
- Create: `spam-harm/app/test_predict.py`

**Interfaces:**
- Consumes: `spam-harm/iterate/final_model.joblib` from Task 7
- Produces: `predict_message(text: str) -> dict` returning `{"label": "spam"/"ham", "confidence": float, "probabilities": {"ham": float, "spam": float}}`. Consumed by Task 9 (`app.py`) and Task 10 (`stress_test.py`).

- [ ] **Step 1: Write the failing test**

Create `spam-harm/app/test_predict.py`:

```python
"""
spam-harm/app/test_predict.py

Confirms predict_message() works on normal input and raises loudly on
invalid input rather than silently producing a bogus prediction --
the same "silent-wrong-answer bugs are worse than crashes" lesson from
the Disney+ project's stress_test.py.

Run:
    python test_predict.py
"""

from predict import predict_message


def run_checks():
    spam_result = predict_message("WINNER!! You have been selected to receive a "
                                   "£1000 cash prize. Call 09061701461 now to claim!")
    ham_result = predict_message("Hey, are we still on for lunch tomorrow at noon?")

    checks = [
        ("spam-like message returns a dict with 'label'", "label" in spam_result),
        ("spam-like message classified as spam", spam_result["label"] == "spam"),
        ("ham-like message classified as ham", ham_result["label"] == "ham"),
        ("confidence is between 0 and 1", 0.0 <= spam_result["confidence"] <= 1.0),
        ("probabilities has both classes", set(spam_result["probabilities"].keys()) == {"ham", "spam"}),
    ]

    error_checks = []
    for name, bad_call in [
        ("non-string input raises TypeError", lambda: predict_message(12345)),
        ("None input raises TypeError", lambda: predict_message(None)),
        ("empty string raises ValueError", lambda: predict_message("")),
        ("whitespace-only string raises ValueError", lambda: predict_message("   ")),
    ]:
        try:
            bad_call()
            error_checks.append((name, False))
        except (TypeError, ValueError):
            error_checks.append((name, True))
        except Exception:
            error_checks.append((name, False))

    all_checks = checks + error_checks
    print(f"Running {len(all_checks)} checks...\n")
    failed = 0
    for name, passed in all_checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(all_checks) - failed}/{len(all_checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd spam-harm/app && python test_predict.py`
Expected: `ModuleNotFoundError: No module named 'predict'`

- [ ] **Step 3: Write minimal implementation**

Create `spam-harm/app/predict.py`:

```python
"""
spam-harm/app/predict.py

Wraps the final spam/harm model (a full TF-IDF + classifier Pipeline)
in a single predict function that takes a raw message string and
returns a prediction. The model does its own vectorization internally
(it's a Pipeline), so there's no separate fitted-vectorizer to manage
here, unlike the Disney+ project's predict.py.

Run directly for a smoke test:
    python predict.py
"""

import joblib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_PATH = REPO_ROOT / "iterate" / "final_model.joblib"

_model = None


def _load():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _validate_input(text):
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}: {text!r}")
    if not text.strip():
        raise ValueError("text must not be empty or whitespace-only")


def predict_message(text: str) -> dict:
    """
    text: the raw message to classify.
    Returns: {"label": "spam"/"ham", "confidence": float, "probabilities": {"ham": float, "spam": float}}
    Raises: TypeError/ValueError for invalid input -- deliberately loud,
            since a silent wrong answer is worse than a crash.
    """
    _validate_input(text)
    model = _load()

    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0]  # index 0 = ham, index 1 = spam (class order 0,1)

    label = "spam" if pred == 1 else "ham"
    probabilities = {"ham": float(proba[0]), "spam": float(proba[1])}
    confidence = probabilities[label]

    return {"label": label, "confidence": confidence, "probabilities": probabilities}


if __name__ == "__main__":
    print("Spam example:")
    print(predict_message("WINNER!! You have been selected to receive a £1000 "
                           "cash prize. Call 09061701461 now to claim!"))
    print("\nHam example:")
    print(predict_message("Hey, are we still on for lunch tomorrow at noon?"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd spam-harm/app && python test_predict.py`
Expected: `9/9 passed.`

- [ ] **Step 5: Commit**

```bash
git add spam-harm/app/predict.py spam-harm/app/test_predict.py
git commit -m "feat(spam-harm): add predict_message() inference wrapper with input validation"
```

---

### Task 9: Streamlit app

**Files:**
- Create: `spam-harm/app/app.py`

**Interfaces:**
- Consumes: `predict_message(text: str) -> dict` from Task 8

- [ ] **Step 1: Write the app**

Create `spam-harm/app/app.py`:

```python
"""
spam-harm/app/app.py

Streamlit interface for the spam/harm text classifier. Single text box
in, prediction out -- try/except around the predict call so a crash
never reaches the user as a raw traceback (same pattern as the Disney+
project's app.py).

Run:
    streamlit run app.py
"""

import streamlit as st
from predict import predict_message

st.set_page_config(page_title="Spam/Harm Text Classifier", page_icon="🚫")

st.title("🚫 Spam/Harm Text Classifier")
st.caption(
    "Classifies a message as spam or ham (legitimate) using a TF-IDF + "
    "classical ML model trained on a small labeled spam/ham dataset. "
    "See README.md for accuracy and known limitations before trusting a prediction."
)

message = st.text_area("Paste a message to classify", height=120)

if st.button("Classify", type="primary"):
    if not message.strip():
        st.warning("Enter a message first.")
    else:
        try:
            result = predict_message(message)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("This shouldn't happen with valid input -- if you hit this, please note exactly what you entered.")
        else:
            label = result["label"]
            confidence = result["confidence"]
            if label == "spam":
                st.error(f"Predicted: **SPAM** (confidence: {confidence:.0%})")
            else:
                st.success(f"Predicted: **HAM** (legitimate) (confidence: {confidence:.0%})")

            st.write("Probability breakdown:")
            st.bar_chart(result["probabilities"])

            if confidence < 0.65:
                st.info(
                    "Low confidence -- the model is genuinely unsure here. "
                    "See README.md for known limitations."
                )
```

- [ ] **Step 2: Manual smoke test**

Run: `cd spam-harm/app && streamlit run app.py`

Confirm in the browser:
1. Paste `"WINNER!! You have been selected to receive a £1000 cash prize. Call 09061701461 now to claim!"` → click Classify → expect **SPAM**.
2. Paste `"Hey, are we still on for lunch tomorrow at noon?"` → click Classify → expect **HAM**.
3. Click Classify with the box empty → expect the "Enter a message first" warning, no crash.

- [ ] **Step 3: Commit**

```bash
git add spam-harm/app/app.py
git commit -m "feat(spam-harm): add Streamlit app for interactive spam/ham classification"
```

---

### Task 10: Stress testing

**Files:**
- Create: `spam-harm/app/stress_test.py`

**Interfaces:**
- Consumes: `predict_message(text: str) -> dict` from Task 8

- [ ] **Step 1: Write the stress test**

Create `spam-harm/app/stress_test.py`:

```python
"""
spam-harm/app/stress_test.py

Adversarial/edge-case inputs for predict_message(), same philosophy as
the Disney+ project's stress_test.py: silent-wrong-answer bugs are
worse than crashes, so this exists to catch both categories before a
live demo does.

Run:
    python stress_test.py
"""

from predict import predict_message

# (name, input, expected: "ok" or the exception type expected to be raised)
CASES = [
    ("normal ham message", "Hey, are we still on for lunch tomorrow?", "ok"),
    ("normal spam message", "WINNER!! Claim your free prize now, call 09061701461", "ok"),
    ("very long message (2000 chars)", "free prize call now! " * 100, "ok"),
    ("single character", "k", "ok"),
    ("emoji-only message", "😀😀😀🎉🎉", "ok"),
    ("non-English text", "Bonjour, comment ça va aujourd'hui?", "ok"),
    ("message with only numbers", "1234567890", "ok"),
    ("message with unusual whitespace", "  hey\tthere\n\nhow are you  ", "ok"),

    ("empty string", "", ValueError),
    ("whitespace-only string", "   ", ValueError),
    ("None input", None, TypeError),
    ("integer input", 12345, TypeError),
    ("list input", ["hey there"], TypeError),
]


def run():
    passed, failed = 0, 0
    for name, text, expected in CASES:
        try:
            result = predict_message(text)
            if expected == "ok":
                print(f"PASS | {name:42s} -> {result['label']} ({result['confidence']:.0%})")
                passed += 1
            else:
                print(f"FAIL | {name:42s} -> expected {expected.__name__}, but got a result instead: {result['label']}")
                failed += 1
        except Exception as e:
            if expected != "ok" and isinstance(e, expected):
                print(f"PASS | {name:42s} -> correctly raised {type(e).__name__}: {e}")
                passed += 1
            else:
                print(f"FAIL | {name:42s} -> raised {type(e).__name__}: {e} (expected {expected})")
                failed += 1

    print(f"\n{passed}/{passed + failed} passed.")
    if failed:
        raise SystemExit(f"{failed} case(s) failed.")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run it**

Run: `cd spam-harm/app && python stress_test.py`
Expected: `13/13 passed.` If any case fails, add explicit validation to `predict.py`'s `_validate_input()` for that case (following Task 8's pattern) before proceeding — do not weaken the test to make it pass.

- [ ] **Step 3: Commit**

```bash
git add spam-harm/app/stress_test.py
git commit -m "test(spam-harm): add adversarial stress tests for predict_message()"
```

---

### Task 11: Packaging, deployment guide, and final README

**Files:**
- Create: `spam-harm/app/requirements.txt`
- Create: `spam-harm/app/DEPLOYMENT.md`
- Create: `spam-harm/README.md`

**Interfaces:** none (this is documentation + a manual deployment step)

- [ ] **Step 1: Pin requirements to what's actually installed locally**

Run this to capture exact local versions (do not guess versions — use what's actually installed, per the lesson learned deploying the Disney+ app):

```bash
python -c "import streamlit, sklearn, pandas, numpy, joblib, matplotlib, seaborn; print('streamlit==' + streamlit.__version__); print('scikit-learn==' + sklearn.__version__); print('pandas==' + pandas.__version__); print('numpy==' + numpy.__version__); print('joblib==' + joblib.__version__); print('matplotlib==' + matplotlib.__version__); print('seaborn==' + seaborn.__version__)"
```

Create `spam-harm/app/requirements.txt` with those exact versions, one per line (e.g. `streamlit==1.59.2`, etc. — substitute the real output from the command above).

- [ ] **Step 2: Write the deployment guide**

Create `spam-harm/app/DEPLOYMENT.md`:

```markdown
# Deployment Guide (Streamlit Community Cloud)

## Before you start

This app needs `spam-harm/iterate/final_model.joblib` committed to the
repo — Streamlit Cloud only sees what's on GitHub, not your local
machine. Confirm it's tracked (not gitignored):

```bash
git ls-files spam-harm/iterate/final_model.joblib
```

If that prints nothing, force-add it: `git add -f spam-harm/iterate/final_model.joblib`.

## Steps

1. Push everything to GitHub, including `spam-harm/` and `final_model.joblib`.
2. Go to **share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**.
4. Fill in:
   - **Repository:** `LacadenJulianna/nlp-text-classifier-pipeline`
   - **Branch:** `main`
   - **Main file path:** `spam-harm/app/app.py`
5. Click **Deploy**.
6. **Once live, verify it against the exact documented smoke-test input
   before calling this "deployed"** — this is a hard-learned rule from
   the Disney+ deployment, not optional:
   - Paste: `WINNER!! You have been selected to receive a £1000 cash
     prize. Call 09061701461 now to claim!` → expect **SPAM**.
   - Paste: `Hey, are we still on for lunch tomorrow at noon?` → expect
     **HAM**.
   - If either doesn't match, do NOT assume the environment is broken —
     first confirm you used the *exact* input above, not a paraphrase
     or a different example. Only chase environment/dependency
     mismatches once the exact input is confirmed and still disagrees.

## If it fails to deploy

Check the Streamlit Cloud deploy logs (Manage app dashboard):
- `FileNotFoundError` on the model file → it wasn't actually committed (see "Before you start").
- A version-related unpickling error → the installed versions didn't match `requirements.txt`; check what the log says actually got installed.

## The public URL

Save the `https://<something>.streamlit.app` URL once live — it goes in `README.md` below.
```

- [ ] **Step 3: Write the final README**

Create `spam-harm/README.md`:

```markdown
# Spam/Harm Text Classifier

Classifies a text message as spam or ham (legitimate) using TF-IDF +
classical ML (Naive Bayes baseline, tuned Logistic Regression as the
final model — whichever won on held-out spam F1, see
`iterate/model_choice.md`). Trained on a small labeled spam/ham dataset
(1,000 raw rows; 677 after removing exact-duplicate messages).

**Live demo:** [add your Streamlit Cloud URL here once deployed — see app/DEPLOYMENT.md]

## Setup (running locally)

```bash
cd spam-harm/app
pip install -r requirements.txt
streamlit run app.py
```

Requires `spam-harm/iterate/final_model.joblib` to exist (committed to
this repo directly, produced by `iterate/evaluate_candidate.py`).

## Usage

1. Paste any message into the text box.
2. Click **Classify**.
3. See the predicted label (spam/ham), a confidence score, and the full
   probability breakdown.

## What this model can do

- Trained on 677 labeled messages (after deduplication), evaluated on a held-out test set.
- Headline metric is **spam-class F1** (see `iterate/model_choice.md`
  for the exact number) — not accuracy, since the deduplicated dataset is
  ~74% ham and accuracy alone would reward a model that just says "ham" every time.
- Beats the naive baseline (always predict ham, spam F1 = 0.0) by a
  real, verified margin (see `baseline/metrics_report.md`).

## What this model can't do — read this before trusting a prediction

- Trained on a small (677-message, post-dedup), possibly synthetic
  dataset of longer email-style text, not real-world SMS traffic —
  performance on genuinely different text (short SMS, social media
  DMs, etc.) is untested.
- English only — the training data is English-language text.
- "Harm" here means spam/unsolicited-commercial text specifically, not
  a general toxicity/harassment classifier — those are a different
  labeling task with different training data.
- Doesn't use stemming (see `iterate/tuning_log.md` for why this was
  discarded for this project's scope) — messages with heavy misspelling
  or unusual word forms may be harder to classify correctly.

## Project structure

```
data/       Raw dataset, cleaning pipeline, canonical train/test split
eda/        Class balance, message length, and token-frequency exploration
baseline/   TF-IDF + Naive Bayes baseline model and metrics
iterate/    Tuned Logistic Regression candidate, comparison, final model
app/        predict.py, Streamlit app, stress tests, deployment guide
```

## Known open items (not yet resolved)

- Dataset provenance/license unconfirmed — the source of `spam-harm/data/raw/spam.csv` (downloaded from Kaggle to `C:\Users\Julianna\Downloads\spam_dataset.csv`) and whether it's a real or synthetic dataset was not verified. Confirm before any use beyond this learning project.
- Stemming was considered and discarded for this iteration (see `iterate/tuning_log.md`) — worth revisiting if a future iteration shows token sparsity is a real bottleneck.
```

- [ ] **Step 4: Deploy and verify (manual — see `spam-harm/app/DEPLOYMENT.md`)**

Follow the steps in `spam-harm/app/DEPLOYMENT.md` to push to GitHub and deploy on Streamlit Community Cloud. Verify against the exact smoke-test input specified there. Once confirmed, fill in the live URL in both `spam-harm/README.md` and `spam-harm/app/DEPLOYMENT.md`.

- [ ] **Step 5: Commit**

```bash
git add spam-harm/app/requirements.txt spam-harm/app/DEPLOYMENT.md spam-harm/README.md
git commit -m "docs(spam-harm): add requirements, deployment guide, and final README"
```

---

## Self-Review Notes

- **Spec coverage:** every section of the design spec maps to a task — data/cleaning (Tasks 1-2), EDA (Task 3), baseline (Tasks 4-5), iteration incl. stopword/stemming question (Task 6), model selection (Task 7), packaging (Task 8), Streamlit app (Task 9), stress test (Task 10), deployment/README (Task 11).
- **Deviation from spec, noted explicitly:** the spec's `iterate/tune_final.py` is folded into `train_candidate.py`'s single `GridSearchCV` call (Task 6) rather than a separate script, since the grid search already produces the tuning result directly — avoids a redundant second training pass. `feature_pipeline.py`-style separate vectorizer artifact is intentionally not created — each model is a single `Pipeline` object bundling vectorizer + classifier, which is more idiomatic sklearn for text and simpler for `predict.py` to consume.
- **Type consistency check:** `predict_message()`'s return shape (`{"label", "confidence", "probabilities"}`) is defined once in Task 8 and used identically in Task 9 (`app.py`) and Task 10 (`stress_test.py`) — no drift. `final_model.joblib`'s path (`spam-harm/iterate/final_model.joblib`) is referenced identically in Task 8's `MODEL_PATH` and Task 11's `DEPLOYMENT.md`.
