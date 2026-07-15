"""
stage04/stress_test.py

Week 4, Day 20: Stress-test it

Two real bugs were found running this against the original predict.py,
both silent-wrong-answer bugs rather than crashes -- arguably worse,
since nothing flagged them as errors:

  1. release_year=None silently produced a plausible-looking prediction
     built from a NaN input that should never have reached the model.
  2. is_movie='False' (a string) is truthy in Python -- it was silently
     treated as is_movie=True, flipping duration_value's meaning from
     "seasons" to "minutes" and producing a different, wrong prediction
     with no error anywhere.

Both are now caught by explicit validation in predict.py. This test
file exists so that fix doesn't silently regress later.

Run:
    python stress_test.py
"""

from predict import predict_rating

ALL_GENRES = [
    "Family", "Animation", "Comedy", "Action-Adventure", "Coming of Age",
    "Animals & Nature", "Fantasy", "Documentary", "Drama", "Kids",
    "Docuseries", "Science Fiction", "Historical", "Music", "Biographical",
]

# (name, kwargs, expected: "ok" or the exception type expected to be raised)
CASES = [
    ("empty genre list", dict(genres=[], release_year=2020, duration_value=90, is_movie=True), "ok"),
    ("unknown genre string -> Other bucket", dict(genres=["Sports Documentary"], release_year=2020, duration_value=90, is_movie=True), "ok"),
    ("duplicate genres", dict(genres=["Comedy", "Comedy", "Comedy"], release_year=2020, duration_value=90, is_movie=True), "ok"),
    ("all 15 known genres at once", dict(genres=ALL_GENRES, release_year=2020, duration_value=90, is_movie=True), "ok"),
    ("float release_year, whole number", dict(genres=["Comedy"], release_year=2020.0, duration_value=90, is_movie=True), "ok"),

    ("negative duration", dict(genres=["Comedy"], release_year=2020, duration_value=-10, is_movie=True), ValueError),
    ("duration zero", dict(genres=["Comedy"], release_year=2020, duration_value=0, is_movie=True), ValueError),
    ("implausible movie duration", dict(genres=["Comedy"], release_year=2020, duration_value=999999, is_movie=True), ValueError),
    ("implausible TV season count", dict(genres=["Comedy"], release_year=2020, duration_value=999, is_movie=False), ValueError),
    ("release_year out of range (past)", dict(genres=["Comedy"], release_year=-500, duration_value=90, is_movie=True), ValueError),
    ("release_year out of range (future)", dict(genres=["Comedy"], release_year=9999, duration_value=90, is_movie=True), ValueError),
    ("release_year is None", dict(genres=["Comedy"], release_year=None, duration_value=90, is_movie=True), TypeError),
    ("duration_value is None", dict(genres=["Comedy"], release_year=2020, duration_value=None, is_movie=True), TypeError),
    ("is_movie is a string, not bool", dict(genres=["Comedy"], release_year=2020, duration_value=90, is_movie="False"), TypeError),
    ("genre list contains empty string", dict(genres=[""], release_year=2020, duration_value=90, is_movie=True), TypeError),
    ("genre list contains None", dict(genres=[None], release_year=2020, duration_value=90, is_movie=True), TypeError),
]


def run():
    passed, failed = 0, 0
    for name, kwargs, expected in CASES:
        try:
            result = predict_rating(**kwargs)
            if expected == "ok":
                print(f"PASS | {name:42s} -> {result['rating']} ({result['confidence']:.0%})")
                passed += 1
            else:
                print(f"FAIL | {name:42s} -> expected {expected.__name__}, but got a result instead: {result['rating']}")
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
