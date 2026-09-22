"""Unit tests for src/preprocessing.py.

Run with: python -m unittest discover tests
(also runs under pytest, if installed)
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import preprocessing  # noqa: E402


class CleanTextTest(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(preprocessing.clean_text("HELLO World"), "hello world")

    def test_removes_urls(self):
        self.assertEqual(
            preprocessing.clean_text("Visit http://example.com now www.foo.bar"),
            "visit now",
        )

    def test_removes_numbers_and_punctuation(self):
        self.assertEqual(preprocessing.clean_text("Call 1800-FREE!! now!!"), "call free now")

    def test_collapses_whitespace(self):
        self.assertEqual(preprocessing.clean_text("  lots   of   spaces \n\t"), "lots of spaces")


class LoadSmsDataTest(unittest.TestCase):
    def test_label_message_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sms.csv"
            pd.DataFrame({"label": ["ham", "spam"], "message": ["hello", "win free"]}).to_csv(path, index=False)
            df = preprocessing.load_sms_data(path)
            self.assertEqual(list(df.columns), ["label", "message"])
            self.assertEqual(list(df["label"]), [0, 1])

    def test_v1_v2_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sms.csv"
            pd.DataFrame({"v1": ["spam", "ham"], "v2": ["buy now", "see you"]}).to_csv(path, index=False)
            df = preprocessing.load_sms_data(path)
            self.assertEqual(list(df.columns), ["label", "message"])
            self.assertEqual(list(df["label"]), [1, 0])
            self.assertEqual(list(df["message"]), ["buy now", "see you"])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            preprocessing.load_sms_data(Path("/nonexistent/sms.csv"))


class LoadEmailDataTest(unittest.TestCase):
    def test_subject_message_spamham_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emails.csv"
            pd.DataFrame(
                {
                    "Subject": ["Hello", "Limited offer"],
                    "Message": ["How are you?", "Claim your prize"],
                    "Spam/Ham": ["ham", "spam"],
                }
            ).to_csv(path, index=False)
            df = preprocessing.load_email_data(path)
            self.assertEqual(list(df.columns), ["label", "message"])
            self.assertEqual(list(df["label"]), [0, 1])
            self.assertEqual(
                list(df["message"]),
                ["Hello How are you?", "Limited offer Claim your prize"],
            )

    def test_prenormalized_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emails.csv"
            pd.DataFrame({"label": ["ham"], "message": ["just text"]}).to_csv(path, index=False)
            df = preprocessing.load_email_data(path)
            self.assertEqual(list(df["label"]), [0])
            self.assertEqual(list(df["message"]), ["just text"])

    def test_missing_label_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "emails.csv"
            pd.DataFrame({"Subject": ["hi"], "Message": ["there"]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                preprocessing.load_email_data(path)


class LoadRawDataTest(unittest.TestCase):
    def test_auto_detect_email_by_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "email_data.csv"
            pd.DataFrame(
                {"Subject": ["Hi"], "Message": ["Hey"], "Spam/Ham": ["ham"]}
            ).to_csv(path, index=False)
            df = preprocessing.load_raw_data(path)
            self.assertEqual(list(df["label"]), [0])

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            preprocessing.load_raw_data(format="bogus")


class PreprocessDataTest(unittest.TestCase):
    def test_adds_cleaned_message(self):
        df = pd.DataFrame({"label": [0], "message": ["Call 1800-FREE!! Now"]})
        out = preprocessing.preprocess_data(df)
        self.assertIn("cleaned_message", out.columns)
        self.assertEqual(out["cleaned_message"].iloc[0], "call free now")

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"label": [0], "message": ["Hello"]})
        preprocessing.preprocess_data(df)
        self.assertNotIn("cleaned_message", df.columns)


class SaveProcessedDataTest(unittest.TestCase):
    def test_saves_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = preprocessing.PROCESSED_DATA_DIR
            preprocessing.PROCESSED_DATA_DIR = Path(tmp)
            try:
                df = pd.DataFrame({"label": [0], "cleaned_message": ["hi"]})
                out = preprocessing.save_processed_data(df, "test_out.csv")
                self.assertTrue(out.exists())
                loaded = pd.read_csv(out)
                self.assertEqual(list(loaded.columns), ["label", "cleaned_message"])
            finally:
                preprocessing.PROCESSED_DATA_DIR = old_dir


class LeetspeakDecodingTest(unittest.TestCase):
    """Spammers swap letters for digits; clean_text has to undo it.

    Digits used to be deleted outright, which destroyed the obfuscation instead
    of normalising it ("fr33" became "fr"), so obfuscated spam slipped through.
    """

    def test_decodes_common_substitutions(self):
        self.assertEqual(preprocessing.clean_text("fr33 m0ney cl1ck n0w"), "free money click now")

    def test_decodes_inside_longer_words(self):
        self.assertEqual(
            preprocessing.clean_text("C0NGRATULATIONS U h4ve w0n"),
            "congratulations u have won",
        )

    def test_decodes_at_and_dollar(self):
        self.assertEqual(preprocessing.clean_text("c@sh and m0ney"), "cash and money")

    def test_leaves_standalone_numbers_alone(self):
        """Phone numbers and order ids must not be rewritten into letters."""
        self.assertEqual(preprocessing.clean_text("order 12345 shipped"), "order shipped")

    def test_leaves_plain_words_alone(self):
        self.assertEqual(preprocessing.clean_text("free money now"), "free money now")

    def test_digits_only_becomes_empty(self):
        self.assertEqual(preprocessing.clean_text("12345 67890"), "")


class UnicodeNormalizationTest(unittest.TestCase):
    def test_fullwidth_letters_fold_to_ascii(self):
        """NFKC must defeat full-width look-alike evasion."""
        self.assertEqual(
            preprocessing.clean_text("ＦＲＥＥ ＭＯＮＥＹ ＣＬＩＣＫ ＮＯＷ"),
            "free money click now",
        )

    def test_combining_marks_are_dropped_known_limitation(self):
        r"""Known limitation, pinned so it cannot change by accident.

        [^\w\s] removes category-Mn combining marks, so Devanagari vowel signs
        are lost. Preserving them was measured and made the model slightly worse
        on this dataset, so the simpler behaviour is intentional. The detector is
        trained on English SMS/email only.
        """
        self.assertEqual(preprocessing.clean_text("नमस्ते"), "नमस त")


class CleanTextTypeTest(unittest.TestCase):
    def test_rejects_non_string(self):
        for value in (None, 123, ["a"], {"a": 1}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    preprocessing.clean_text(value)


if __name__ == "__main__":
    unittest.main()