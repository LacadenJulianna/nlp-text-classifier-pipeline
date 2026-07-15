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
        # Safe to load: MODEL_PATH points to final_model.joblib, an artifact
        # trained internally by this pipeline (trusted, not an untrusted external source)
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
    print(predict_message("This is not a scam, click now to see the details! "
                           "Limited time offer, act now! Get instant access to premium services. "
                           "For more details, visit our website or contact us directly."))
    print("\nHam example:")
    print(predict_message("Hey, are we still on for lunch tomorrow at noon?"))
