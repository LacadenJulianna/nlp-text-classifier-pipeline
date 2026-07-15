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
    ("normal spam message", "This is not a scam, click now to see the details! Limited time offer, act now! Get instant access to premium services. For more details, visit our website or contact us directly.", "ok"),
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
