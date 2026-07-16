"""Loads the trained IMDB sentiment pipeline and exposes predict_sentiment()."""
import re
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent.parent / "final_model.joblib"
HTML_TAG_RE = re.compile(r"<[^>]+>")

_model = None


def _get_model():
    global _model
    if _model is None:
        # Safe: loads our own final_model.joblib, trained by train.py and
        # committed to this repo — not an artifact from an untrusted source.
        _model = joblib.load(MODEL_PATH)
    return _model


def _clean(text: str) -> str:
    text = HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict_sentiment(review: str) -> dict:
    if not isinstance(review, str) or not review.strip():
        raise ValueError("review must be a non-empty string")

    cleaned = _clean(review)
    if not cleaned:
        raise ValueError("review contains no usable text after cleaning")

    model = _get_model()
    label = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    confidence = float(proba[classes.index(label)])

    return {"sentiment": label, "confidence": confidence}


if __name__ == "__main__":
    example = "This movie was a stunning, beautifully acted masterpiece. I loved every minute of it."
    print(predict_sentiment(example))
