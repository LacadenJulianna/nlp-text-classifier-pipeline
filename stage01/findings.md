# Day 5 — Presentation Notes (10-Minute Demo)

Structured for roughly: 1 min intro, 2 min dataset overview, 2 min per insight (x3), 1 min open questions/close. Adjust live based on questions.

---

## Intro (~1 min)

**Project:** Predict a Disney+ title's content rating (TV-G, PG, TV-14, etc.) from its genre, runtime, release year, and country.

**Why it matters:** streaming platforms need consistent content ratings for parental controls and catalog search. Ratings are sometimes missing or inconsistent when new titles get added — a model that predicts a likely rating from genre/runtime could flag mis-tagged titles for human review, instead of someone re-watching every title by hand.

---

## Dataset overview (~2 min)

- **Source:** Disney+ Movies and TV Shows dataset (Kaggle) — 1,450 titles, 12 original columns.
- **After cleaning (Stage 01):** 1,447 titles — dropped 3 rows with no rating (the target), filled missing `country` with "Unknown," split `duration` into a numeric value + unit (minutes vs. seasons), split `listed_in` into individual genre tags.
- **The target:** `rating`, 9 classes — TV-G and TV-PG are the largest (300+ titles each), TV-Y7-FV is the smallest (13 titles).

---

## Insight 1 — The target is meaningfully imbalanced (~2 min)

TV-G (318) and TV-PG (301) titles outnumber TV-Y7-FV (13) by roughly 24x. Show Chart 1 (rating counts) here.

**Why it matters for the project:** raw accuracy will mislead once modeling starts — a model that only ever predicts "TV-G" would already look deceptively decent. Evaluation needs to be per-class (precision/recall by rating) or use a balanced accuracy score, decided now rather than after a model's already been trained on the wrong assumption.

## Insight 2 — Genre is the strongest signal found so far (~2 min)

Rating distribution differs clearly by genre. Show Chart 6 (genre × rating heatmap) here.

- Kid-facing genres cluster toward G/TV-G: Coming of Age is 51% TV-G, Animation is 53% G+TV-G.
- Documentary and Animals & Nature both lean toward **TV-PG** (58% and 48%) — not a stricter/more mature rating, despite "documentary" sounding that way.
- Action-Adventure has no dominant peak — spread fairly evenly across G/PG/TV-Y7 — a reminder that not every genre will carry equally strong signal.

**Why it matters:** this is the concrete evidence that the project's premise (genre → rating) actually holds in the data, not just in theory.

## Insight 3 — `duration` is entangled with `type`, not just `rating` (~2 min)

`duration_unit` (minutes vs. seasons) predicts `type` (Movie vs. TV Show) with 100% accuracy — every Movie is "min," every TV Show is "Season." Show Chart 4 (duration split) here.

**Why it matters:** `duration` is still a legitimate feature for predicting `rating` this week, but it's effectively acting as a disguised copy of `type`. If a future version of this project ever targets `type` instead of `rating`, `duration` has to be dropped first — otherwise the "model" is just reading the label, not predicting anything.

---

## Open questions (~1 min)

- Dataset license is unconfirmed — needs resolving before this work is shared publicly (flagged since Stage 01).
- Is `country` worth keeping given how dominant "United States" (69.5%) and "Unknown" (14.9%) are?
- How many genres should become individual model features vs. get bucketed into "Other"?
- Should TV-Y7-FV (13 titles) be merged into TV-Y7 given how little data it has on its own?

---

### If asked "what's next"
Stage 02+ (modeling): encode genres (top-N one-hot + "Other" bucket), decide on `country`'s inclusion, pick an evaluation metric that accounts for class imbalance, and establish a baseline (e.g., predict the majority class) before trying anything more complex — so there's a number to actually beat.
