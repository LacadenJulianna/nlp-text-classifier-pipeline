# Day 10 — Presentation Notes: Baseline Model Demo

Structured for ~10 minutes: 1 min recap, 2 min the number, 4 min what it means, 2 min limitations, 1 min next steps.

---

## Recap (~1 min)

Task: predict Disney+ content rating (9 classes) from genre, duration, and release year. Day 6 set the framing and picked macro F1 as the success metric — not accuracy, because with a ~24x gap between the largest and smallest rating class, accuracy can look fine while a model completely ignores rare classes.

## The number (~2 min)

**Naive baseline (always guess the majority class, TV-G):** macro F1 = 0.040
**Logistic regression baseline (Day 8 model):** macro F1 = **0.527**

That's a **13.2x improvement** over doing nothing. In plain terms: a model that always guesses the most common rating is nearly useless once you check its performance class-by-class (0.040 out of 1.0) — this model, trained on genre/duration/release year alone, is actually learning real patterns, not just memorizing the majority class.

Accuracy for context: 61.7% — but leading with macro F1 rather than this number is the point of Day 6's decision, not a footnote.

## What it means, class by class (~4 min)

Show the confusion matrix here.

- **Strongest:** PG-13 (0.75 F1, though only 13 test examples — small sample, don't over-claim this), TV-Y7 (0.69), TV-G (0.67).
- **Weakest, and why it matters:** TV-14 scored only 0.26 F1. Checking the confusion matrix directly: of 16 actual TV-14 titles, 9 were predicted as **TV-PG** — its immediate neighbor on the TV rating scale (TV-Y7 < TV-G < TV-PG < TV-14). This connects directly back to the limitation flagged in Day 6: plain multi-class classification treats every wrong answer as equally wrong, but this model's mistakes aren't random — they cluster on adjacent rating tiers. That's evidence the ordinal structure we chose not to model is actually costing real performance, not just a theoretical concern.
- **TV-Y7-FV scored 0.00 F1** — expected, not a modeling failure: only 3 test examples exist for this class at all. No baseline model can learn a reliable pattern from that little data. Worth stating plainly in the demo rather than treating it as a bug to explain away.

## Limitations (~2 min)

- **Sample size on rare classes is a hard floor**, not something more model complexity fixes. TV-Y7-FV (13 total titles) may need to be merged into TV-Y7 or excluded entirely — flagged as an open question since Day 4/5, still unresolved.
- **`country` was excluded** from this baseline (Day 6 decision: dominated by "United States" + a large "Unknown" bucket). Not tested with it included yet — an open comparison, not a settled conclusion.
- **No class-weight balancing was used on purpose** — this is the simplest reasonable baseline, not a tuned model. `class_weight="balanced"` is a natural next step to test whether it helps the weaker classes (TV-14 especially) without hurting the strong ones.

## Next steps (~1 min)

- Test `class_weight="balanced"` and compare macro F1 directly against 0.527.
- Test with `country` included vs. excluded — settle the open question from Day 6 with an actual number instead of a guess.
- Consider whether an ordinal-aware approach (given the TV-14/TV-PG adjacency finding) is worth prototyping, given it's now backed by evidence, not just a theoretical caveat.
