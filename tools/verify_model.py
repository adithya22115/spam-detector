"""End-to-end verification of the exported spam model.

Part A  Reproduces the exact src/train.py split (test_size=0.2, random_state=42)
        and scores the SHIPPED pickle on that held-out set, then diffs the result
        against results/metrics_all.json.
Part B  Curated battery of obvious spam/ham plus adversarial edge cases
        (leetspeak, full-width look-alikes, empty, and oversized input).

Run:  .venv/Scripts/python.exe tools/verify_model.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import predict  # noqa: E402
from preprocessing import load_email_data, load_sms_data, preprocess_data  # noqa: E402

MODEL_PATH = ROOT / "models" / "spam_model.pkl"
METRICS_PATH = ROOT / "results" / "metrics_all.json"


def _load_split():
    """Rebuild the exact train/test split used by src/train.py."""
    sms = load_sms_data()
    email = load_email_data()
    sms["source"] = "sms"
    email["source"] = "email"
    df = preprocess_data(pd.concat([sms, email], ignore_index=True))
    _, X_test, _, y_test = train_test_split(
        df["cleaned_message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    return df, X_test, y_test


def part_a() -> bool:
    print("=" * 78)
    print("PART A - held-out evaluation of the shipped model")
    print("=" * 78)

    df, X_test, y_test = _load_split()
    source_test = df.loc[X_test.index, "source"]
    print(f"Test set: {len(y_test):,} messages")

    model = joblib.load(MODEL_PATH)
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    elapsed = time.perf_counter() - start

    measured = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_spam": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_test, y_proba)), 4),
    }
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))
    print(f"confusion [[TN FP] [FN TP]] = {confusion_matrix(y_test, y_pred).tolist()}")
    print(f"scored in {elapsed:.1f}s\n")

    print("Per-source:")
    for src in sorted(source_test.unique()):
        mask = (source_test == src).to_numpy()
        print(
            f"  {src:<6} n={mask.sum():>5}  acc={accuracy_score(y_test[mask], y_pred[mask]):.4f}  "
            f"f1_spam={f1_score(y_test[mask], y_pred[mask]):.4f}"
        )

    ok = True
    if METRICS_PATH.exists():
        recorded = json.loads(METRICS_PATH.read_text())
        print(f"\nvs {METRICS_PATH.relative_to(ROOT)}:")
        for key, value in measured.items():
            match = abs(recorded.get(key, -1) - value) < 5e-4
            ok &= match
            print(f"  {key:<8} measured={value:<8} recorded={recorded.get(key)!r:<8} {'OK' if match else 'MISMATCH'}")

    errors = int((y_test != y_pred).sum())
    print(f"\n{errors} errors of {len(y_test):,} ({errors / len(y_test) * 100:.2f}%)")
    return ok


CASES: list[tuple[str, str]] = [
    ("spam", "WINNER!! You have won a 1000 pound prize. Call 09061701461 now to claim."),
    ("spam", "URGENT! Your mobile number has been awarded a 2000 bonus."),
    ("spam", "Congratulations! You've won a free iPhone. Click here to claim."),
    ("spam", "Dear customer, your account will be suspended. Verify your details immediately."),
    ("spam", "You have been selected to receive a $500 gift card. Act now!"),
    # obfuscated - the regex frees these now that digits are decoded, not deleted
    ("spam", "W1N FR33 C4SH N0W!!! C1ick h3re"),
    ("spam", "fr33 m0ney cl1ck n0w"),
    ("spam", "Cl1ck h3re to cl41m y0ur fr33 g1ft"),
    ("spam", "ＦＲＥＥ ＭＯＮＥＹ ＣＬＩＣＫ ＮＯＷ"),
    ("spam", "<html><body><h1>CONGRATULATIONS</h1>Claim your FREE prize now!</body></html>"),
    ("spam", "FREE MONEY CLICK NOW " * 200),
    ("ham", "Are you coming home for dinner tonight?"),
    ("ham", "Sorry, I'll call later"),
    ("ham", "Hi team, please find attached the meeting minutes."),
    ("ham", "Your order has shipped and will arrive Thursday."),
    ("ham", "Reminder: dentist appointment tomorrow at 3pm."),
    ("ham", "The build passed on CI, merging now."),
    ("ham", "Please find the quarterly report attached. " * 200),
    # degenerate input - reported as insufficient rather than guessed at
    ("ham", ""),
    ("ham", "     "),
    ("ham", "!!!???..."),
    ("ham", "12345 67890"),
    ("ham", "http://example.com"),
]


def part_b() -> int:
    print("\n" + "=" * 78)
    print("PART B - curated battery (obvious cases + adversarial edge cases)")
    print("=" * 78)

    failures = 0
    start = time.perf_counter()
    for expected, text in CASES:
        result = predict(text, model_path=MODEL_PATH)
        got = "spam" if result["is_spam"] else "ham"
        ok = got == expected
        failures += not ok
        shown = "".join(c if c.isprintable() else " " for c in text)[:42]
        flag = "ok  " if ok else "FAIL"
        note = " [insufficient]" if result["insufficient_text"] else ""
        print(
            f"[{flag}] expected={expected:<4} got={got:<4} p={result['spam_probability']:.3f} "
            f"{result['confidence']:<6} | {shown}{note}"
        )
    elapsed = time.perf_counter() - start
    print(
        f"\n{len(CASES) - failures}/{len(CASES)} as expected ({elapsed / len(CASES) * 1000:.0f} ms/call; "
        f"the model is cached, so this is inference cost only)"
    )
    return failures


if __name__ == "__main__":
    metrics_match = part_a()
    battery_failures = part_b()
    print("\n" + "=" * 78)
    print(f"VERDICT: metrics reproducible={metrics_match}  battery failures={battery_failures}")
    print("=" * 78)
