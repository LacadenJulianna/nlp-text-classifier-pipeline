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
