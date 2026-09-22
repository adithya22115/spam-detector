"""Regression tests for the SHIPPED model (models/spam_model.pkl).

These assert that the exported artifact still performs on the held-out split that
src/train.py created, so a bad retrain or a preprocessing change cannot silently
ship. The split is reproduced exactly as in src/train.py (test_size=0.2,
random_state=42, stratify on label).

Slow (~25s): this is the only test that touches the real 39k-row datasets and
vectorises the full test set. Skipped automatically if the model is absent.

Run with: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import predict  # noqa: E402
from preprocessing import load_email_data, load_sms_data, preprocess_data  # noqa: E402

MODEL_PATH = ROOT / "models" / "spam_model.pkl"

# Deliberately set below the current scores (acc 0.981, f1 0.979, brier 0.014)
# so they flag real regressions rather than tiny fluctuations.
MIN_ACCURACY = 0.975
MIN_F1_SPAM = 0.970
MAX_BRIER = 0.030
MIN_F1_SPAM_EMAIL = 0.975
MIN_F1_SPAM_SMS = 0.820


class ShippedModelQualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MODEL_PATH.exists():
            raise unittest.SkipTest(f"{MODEL_PATH} not found; run src/train.py first")

        sms = load_sms_data()
        email = load_email_data()
        sms["source"] = "sms"
        email["source"] = "email"
        df = preprocess_data(pd.concat([sms, email], ignore_index=True))

        _, X_test, _, cls.y_test = train_test_split(
            df["cleaned_message"],
            df["label"],
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )
        cls.source_test = df.loc[X_test.index, "source"].to_numpy()

        model = joblib.load(MODEL_PATH)
        cls.y_pred = model.predict(X_test)
        cls.y_proba = model.predict_proba(X_test)[:, 1]

    def test_accuracy_on_held_out_split(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        self.assertGreaterEqual(
            accuracy, MIN_ACCURACY, f"held-out accuracy dropped to {accuracy:.4f}"
        )

    def test_spam_f1_on_held_out_split(self):
        f1 = f1_score(self.y_test, self.y_pred)
        self.assertGreaterEqual(f1, MIN_F1_SPAM, f"held-out spam F1 dropped to {f1:.4f}")

    def test_probabilities_stay_calibrated(self):
        brier = brier_score_loss(self.y_test, self.y_proba)
        self.assertLessEqual(brier, MAX_BRIER, f"Brier score rose to {brier:.4f}")

    def test_per_source_email_f1(self):
        mask = self.source_test == "email"
        f1 = f1_score(self.y_test[mask], self.y_pred[mask])
        self.assertGreaterEqual(f1, MIN_F1_SPAM_EMAIL, f"email spam F1 dropped to {f1:.4f}")

    def test_per_source_sms_f1(self):
        """SMS is the weaker half (0.85); guard it so it cannot silently rot."""
        mask = self.source_test == "sms"
        f1 = f1_score(self.y_test[mask], self.y_pred[mask])
        self.assertGreaterEqual(f1, MIN_F1_SPAM_SMS, f"sms spam F1 dropped to {f1:.4f}")


class ObfuscationHandlingTest(unittest.TestCase):
    """Leetspeak and look-alike evasion must not slip past the shipped model."""

    OBFUSCATED_SPAM = [
        "fr33 m0ney cl1ck n0w",
        "W1N FR33 C4SH N0W!!!",
        "C0NGRATULATIONS! You have w0n a prize",
        "ＦＲＥＥ ＭＯＮＥＹ ＣＬＩＣＫ ＮＯＷ",
        "Cl1ck h3re to cl41m y0ur fr33 g1ft",
    ]

    def test_obfuscated_spam_is_flagged(self):
        for message in self.OBFUSCATED_SPAM:
            with self.subTest(message=message):
                result = predict(message, model_path=MODEL_PATH)
                self.assertTrue(
                    result["is_spam"],
                    f"missed obfuscated spam: {message!r} (p={result['spam_probability']:.3f})",
                )

    def test_plain_ham_is_not_flagged(self):
        for message in (
            "Are you coming home for dinner tonight?",
            "Hi team, please find the meeting minutes attached.",
            "Reminder: dentist appointment tomorrow at 3pm.",
            "The build passed on CI, merging now.",
        ):
            with self.subTest(message=message):
                result = predict(message, model_path=MODEL_PATH)
                self.assertFalse(
                    result["is_spam"],
                    f"false positive on ham: {message!r} (p={result['spam_probability']:.3f})",
                )


if __name__ == "__main__":
    unittest.main()
