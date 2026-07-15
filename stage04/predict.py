"""
stage04/predict.py

Week 4, Day 16: Package your model

Wraps Stage 03's final model in a single predict function that takes RAW
user input (genres, release year, duration) and returns a prediction.

Why inference is kept separate from training: training DECIDES how to
preprocess data (fits the scaler's mean/std, picks which genres count as
"top"). Inference just APPLIES those already-made decisions to new,
unseen input -- it must reuse the exact same fitted scaler (via
.transform(), never .fit()) and the exact same genre vocabulary saved
during training. If a UI tried to reinvent preprocessing independently,
even a small drift from training-time logic would silently produce
wrong predictions with nothing to flag it as an error.

Run directly for a smoke test:
    python predict.py
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
MODEL_PATH = REPO_ROOT / "stage03" / "final_model.joblib"
PIPELINE_PATH = REPO_ROOT / "stage02" / "features" / "feature_pipeline.joblib"

# Loaded once, on first use -- not on import, so importing this module
# elsewhere (e.g. a test file) doesn't require the model files to exist.
_model = None
_scaler = None
_top_genres = None


def _load():
    global _model, _scaler, _top_genres
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        _scaler = pipeline["scaler"]
        _top_genres = pipeline["top_genres"]
    return _model, _scaler, _top_genres


def get_known_genres() -> list[str]:
    """The 15 genres the model actually knows about by name. Anything
    else gets bucketed into 'Other' at prediction time, same as training."""
    _, _, top_genres = _load()
    return list(top_genres)


def _validate_inputs(genres, release_year, duration_value, is_movie):
    """Raises clear errors for bad input instead of silently producing a
    plausible-looking but wrong prediction. Found via Day 20 stress
    testing: None values and a truthy non-bool `is_movie` (e.g. the
    string 'False', which Python treats as truthy) both passed through
    silently before this validation existed."""
    if not isinstance(is_movie, bool):
        raise TypeError(f"is_movie must be a real bool (True/False), got {type(is_movie).__name__}: {is_movie!r}")
    if release_year is None or not isinstance(release_year, (int, float)):
        raise TypeError(f"release_year must be a number, got {type(release_year).__name__}: {release_year!r}")
    if not (1900 <= release_year <= 2030):
        raise ValueError(f"release_year {release_year} is out of the plausible range (1900-2030)")
    if duration_value is None or not isinstance(duration_value, (int, float)):
        raise TypeError(f"duration_value must be a number, got {type(duration_value).__name__}: {duration_value!r}")
    if duration_value <= 0:
        raise ValueError(f"duration_value must be positive, got {duration_value}")
    if is_movie and duration_value > 500:
        raise ValueError(f"duration_value {duration_value} minutes is implausible for a movie")
    if not is_movie and duration_value > 50:
        raise ValueError(f"duration_value {duration_value} seasons is implausible for a TV show")
    if not isinstance(genres, list) or not all(isinstance(g, str) and g for g in genres):
        raise TypeError(f"genres must be a list of non-empty strings, got: {genres!r}")


def predict_rating(genres: list[str], release_year: int, duration_value: int, is_movie: bool) -> dict:
    """
    genres:          list of genre strings. Unknown genres (not in
                      get_known_genres()) are treated as "Other", same
                      as unseen genres were at training time.
    release_year:     e.g. 2016
    duration_value:   minutes if is_movie, seasons if not
    is_movie:         True for Movie, False for TV Show

    Returns: {"rating": str, "confidence": float, "all_probabilities": dict}
    Raises: TypeError/ValueError for invalid input (see _validate_inputs) --
            deliberately loud, since a silent wrong answer is worse than a crash.
    """
    _validate_inputs(genres, release_year, duration_value, is_movie)
    model, scaler, top_genres = _load()

    duration_minutes = duration_value if is_movie else 0
    duration_seasons = duration_value if not is_movie else 0

    numeric = pd.DataFrame(
        [[release_year, duration_minutes, duration_seasons]],
        columns=["release_year", "duration_minutes", "duration_seasons"],
    )
    numeric_scaled = scaler.transform(numeric)  # transform only -- never re-fit at inference time

    genre_flags = [int(g in genres) for g in top_genres]
    genre_other = int(any(g not in top_genres for g in genres))

    X = np.hstack([numeric_scaled, np.array([genre_flags + [genre_other]])])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    classes = model.classes_

    all_probs = {cls: float(p) for cls, p in zip(classes, proba)}
    confidence = float(max(proba))

    return {
        "rating": pred,
        "confidence": confidence,
        "all_probabilities": dict(sorted(all_probs.items(), key=lambda x: -x[1])),
    }


if __name__ == "__main__":
    # Smoke test against a real title from the training data (row 0:
    # "Duck the Halls: A Mickey Mouse Christmas Special" -- Animation,
    # Family, 23 min, 2016, actual rating TV-G).
    result = predict_rating(
        genres=["Animation", "Family"],
        release_year=2016,
        duration_value=23,
        is_movie=True,
    )
    print("Known genres:", get_known_genres())
    print()
    print("Test prediction (expected: close to TV-G, the actual rating):")
    print(result)
