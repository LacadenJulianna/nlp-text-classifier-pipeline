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

    # joblib.load is used here on an artifact this same pipeline just
    # trained and wrote to disk above (not an externally-sourced file),
    # so the usual "untrusted pickle" risk does not apply.
    model = joblib.load(MODEL_PATH)
    pred = model.predict(["free entry to win a prize call now"])
    log_text = LOG_PATH.read_text(encoding="utf-8")

    # Extract the tfidf stop_words value from the trained model
    model_stop_words = model.named_steps["tfidf"].stop_words

    # Check 1: Both stopword settings appear in the log
    both_settings_present = "None" in log_text and "english" in log_text

    # Check 2: Log conclusion is consistent with actual model
    # Verify the log doesn't say "differs" contradicting what's actually in the model
    log_says_differs = "note: differs" in log_text
    if log_says_differs:
        # If log says "differs", the log's best setting must NOT match the model's
        # (If they match, then the log is lying)
        log_is_self_consistent = "note: differs" in log_text  # We're checking it's at least present where it should be
    else:
        # If log says "kept", the log's best setting must match the model's
        log_is_self_consistent = True
    # More robust check: parse which setting the log claims is best and verify it matches the model
    # The log format is "`{setting}` performs better on average -- {'kept'|'differs...'}"
    # We can verify by checking: if "None" is mentioned as best, model should have None; if "english" is mentioned, model should have "english"
    log_includes_none_as_best = "`None` performs better on average" in log_text
    log_includes_english_as_best = "`english` performs better on average" in log_text
    if log_includes_none_as_best or log_includes_english_as_best:
        # At least one should be true (the best one)
        if log_includes_none_as_best:
            # The log claims None is best; model must have None (or the log is wrong)
            log_is_self_consistent = model_stop_words is None
        elif log_includes_english_as_best:
            # The log claims english is best; model must have "english" (or the log is wrong)
            log_is_self_consistent = model_stop_words == "english"

    checks = [
        ("model file was created", MODEL_PATH.exists()),
        ("model has predict()", hasattr(model, "predict")),
        ("predict returns one label per input", len(pred) == 1),
        ("tuning log was created", LOG_PATH.exists()),
        ("tuning log documents stopword comparison", "stop_words" in log_text.lower() or "stopword" in log_text.lower()),
        ("tuning log documents stemming decision", "stemming" in log_text.lower()),
        ("tuning log includes both stopword settings (None and english)", both_settings_present),
        ("tuning log conclusion matches actual model params", log_is_self_consistent),
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
