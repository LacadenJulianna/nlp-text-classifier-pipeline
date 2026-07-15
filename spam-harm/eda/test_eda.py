"""
spam-harm/eda/test_eda.py

Unit tests for eda.py's pure functions, using a small inline fixture
instead of the full dataset -- these are fast, isolated checks of the
computation logic, independent of what's actually in the real data.

Run:
    python test_eda.py
"""

import pandas as pd
from eda import class_balance, message_length_stats, top_tokens

FIXTURE = pd.DataFrame({
    "label": [0, 0, 0, 1, 1],
    "message": [
        "hey are we still on for lunch",
        "call me when you get a chance",
        "see you tomorrow",
        "WINNER call now to claim your free prize",
        "free entry to win a prize call now",
    ],
})


def run_checks():
    balance = class_balance(FIXTURE)
    length_stats = message_length_stats(FIXTURE)
    spam_tokens = top_tokens(FIXTURE, label_value=1, n=3)

    checks = [
        ("class_balance sums to 1.0", abs(balance.sum() - 1.0) < 1e-9),
        ("class_balance has both classes", set(balance.index) == {0, 1}),
        ("length_stats has one row per class", len(length_stats) == 2),
        ("length_stats has a mean column", "mean" in length_stats.columns),
        ("top_tokens returns requested count or fewer", len(spam_tokens) <= 3),
        ("top_tokens finds 'call' among top spam tokens",
         any(tok == "call" for tok, _ in spam_tokens)),
    ]

    print(f"Running {len(checks)} checks...\n")
    failed = 0
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        failed += not passed

    print(f"\n{len(checks) - failed}/{len(checks)} passed.")
    if failed:
        raise SystemExit(f"{failed} check(s) failed.")


if __name__ == "__main__":
    run_checks()
