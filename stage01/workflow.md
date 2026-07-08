# Stage 01 Workflow — Days 2-3

**Project:** Predict Disney+ content rating (TV-G, PG, TV-14, etc.) from genre, runtime, release year, and country.

## Dataset

- **File:** `disney_plus_titles.csv`
- **Shape:** 1,450 titles, 12 columns
- **Columns:** `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`
- **Source:** column structure and row count match the "Disney+ Movies and TV Shows" dataset on Kaggle (`kaggle.com/datasets/shivamb/disney-movies-and-tv-shows`).
- **License:** **not yet confirmed.** Kaggle renders the license tag client-side, so I couldn't pull it via fetch. Open the dataset page directly and copy the license line here before you present — Day 2's brief explicitly requires this, and it's a 30-second check.

**TODO before Friday's demo:** replace this section with the confirmed license and a one-line note on collection date/method once you've checked the source page.

## Day 2 — Python for data + getting your dataset

Ran `day2_load_and_explore()` in `learn-ml.py` (pandas practice only — load, filter, groupby). What it confirmed:

| Check | Result |
|---|---|
| Load | 1,450 rows load cleanly, no parser errors |
| Filter example | 472 of 1,450 titles released 2017 or later |
| Filter example | 1,052 Movies vs. 398 TV Shows |
| Groupby example | Title counts by `rating` — TV-G (318) and TV-PG (301) are the largest classes, TV-Y7-FV (13) is the smallest |
| Groupby example | Average `release_year` by `type` — Movies average 1999, TV Shows average 2013 (TV catalog skews newer) |

Takeaway: the rating classes are imbalanced (25x gap between the biggest and smallest class). That's a real constraint for Day 4/5 modeling — plain accuracy will look artificially good if a model just guesses TV-G every time. Worth flagging in the Friday demo rather than glossing over.

## Day 3 — Data cleaning

Moved cleaning into its own file, `clean_data.py`, rather than leaving it bundled inside `learn-ml.py`. Reasoning: `learn-ml.py` is exploratory scratch work (prints, one-off filters); `clean_data.py` is a reusable pipeline step that Day 4's EDA notebook — and later, model training — should be able to `import clean_data` from, without re-running Day 2's exploration every time. Run standalone with `python clean_data.py`, or import `clean_data(df)` directly.

Decisions made, and why:

| Issue | Decision | Reasoning |
|---|---|---|
| 3 rows missing `rating` | Dropped | `rating` is the prediction target — a row with no label can't be trained on or scored against |
| `director` missing in 473 rows (33%) | Excluded from feature set (column kept, not used) | High missing rate + thousands of unique names (high cardinality) = mostly noise for a first model |
| `cast` missing in 190 rows (13%) | Excluded from feature set (column kept, not used) | Same reasoning as `director` |
| `country` missing in 219 rows (15%) | Filled with `"Unknown"` | Country is a feature we want to keep; "Unknown" is a legitimate category here, not a guess |
| `date_added` missing in 3 rows | Left as-is, converted to datetime | Not used as a feature this week; not worth dropping rows over |
| `duration` mixes two formats (`"23 min"` vs. `"1 Season"`) | Split into `duration_value` (int) + `duration_unit` (`min`/`Season`) | Can't use a mixed-unit string as a numeric feature without separating what's being measured |
| `listed_in` is a comma-separated string of genres | Split into `genre_list` (list column) | One title can have multiple genres; Day 4 will likely turn this into one-hot columns |
| 0 exact duplicate rows | No action needed | Checked with `.duplicated().sum()` — clean |

**Result:** 1,450 → 1,447 rows after cleaning. Output saved to `disney_plus_titles_clean.csv`.

### Known issue, flagged not fixed: `type` vs. `duration` leakage

`duration_unit` perfectly predicts `type` — every Movie has `min`, every TV Show has `Season`, 100% of the time. If a future version of this project pivots to predicting `type` instead of `rating`, `duration` must be dropped from the feature set first, or the "model" is just reading a disguised copy of the label. Not a bug to fix for this week's target (`rating`), but noted here so it doesn't get rediscovered the hard way later.

## Next: Day 4

- Run `clean_data.py` (or `import clean_data` from it) to produce/refresh `disney_plus_titles_clean.csv` — load that into the EDA notebook, not the raw file.
- Turn `genre_list` into one-hot/dummy columns per genre for use as model features.
- Look at rating distribution by genre and by country — likely where the real signal is.
- Keep the class-imbalance note from Day 2 in mind when picking an evaluation metric (accuracy alone will mislead).