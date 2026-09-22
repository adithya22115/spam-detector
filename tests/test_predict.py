"""Unit tests for src/predict.py.

Run with: python -m unittest discover tests
(also runs under pytest, if installed)
"""

import sys
import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from predict import load_model, predict  # noqa: E402


def _tiny_model():
    X = ["win free money now", "congratulations you won", "how are you today", "see you at lunch"]
    y = [1, 1, 0, 0]
    return Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]).fit(X, y)


def _union_model():
    """Production-style pipeline: word+char FeatureUnion + LinearSVC."""
    X = [
        "win free money now", "congratulations you won", "claim your prize",
        "how are you today", "see you at lunch", "thanks for the call",
    ]
    y = [1, 1, 1, 0, 0, 0]
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
        ]
    )
    return Pipeline([("features", features), ("clf", LinearSVC(max_iter=2000))]).fit(X, y)


class LoadModelTest(unittest.TestCase):
    def test_missing_model_raises(self):
        with self.assertRaises(SystemExit):
            load_model(Path("/nonexistent/model.pkl"))

    def test_loads_saved_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            joblib.dump(_tiny_model(), path)
            model = load_model(path)
            self.assertIsNotNone(model)


class PredictTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = Path(self.tmp.name) / "model.pkl"
        joblib.dump(_tiny_model(), self.model_path)

    def test_returns_result_dict(self):
        result = predict("win free money now", model_path=self.model_path)
        self.assertEqual(result["text"], "win free money now")
        self.assertIsInstance(result["is_spam"], bool)
        self.assertGreaterEqual(result["spam_probability"], 0.0)
        self.assertLessEqual(result["spam_probability"], 1.0)
        self.assertIn(result["confidence"], {"high", "medium", "low"})
        self.assertIsInstance(result["top_features"], list)

    def test_classifies_spam_message(self):
        result = predict("Congratulations! You won a free iPhone. Click now!", model_path=self.model_path)
        self.assertTrue(result["is_spam"])
        self.assertGreater(result["spam_probability"], 0.5)

    def test_top_features_structure(self):
        result = predict("win free money now", model_path=self.model_path)
        self.assertTrue(result["top_features"], "expected non-empty top_features")
        for feature in result["top_features"]:
            self.assertIn("word", feature)
            self.assertIn("weight", feature)
            self.assertIn(feature["direction"], {"spam", "ham"})

    def test_classifies_ham_message(self):
        result = predict("how are you today", model_path=self.model_path)
        self.assertFalse(result["is_spam"])

    def test_decision_function_model(self):
        """LinearSVC has no predict_proba; probability comes from decision_function."""
        model = Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", LinearSVC(max_iter=2000))]
        ).fit(
            ["win free money", "you won now", "how are you", "see you later"],
            [1, 1, 0, 0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "svc.pkl"
            joblib.dump(model, path)
            result = predict("win free money now", model_path=path)
        self.assertIsInstance(result["is_spam"], bool)
        self.assertGreaterEqual(result["spam_probability"], 0.0)
        self.assertLessEqual(result["spam_probability"], 1.0)
        self.assertIn(result["confidence"], {"high", "medium", "low"})

    def test_union_pipeline_model(self):
        """Word+char FeatureUnion (production layout): predictions + top features."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "union.pkl"
            joblib.dump(_union_model(), path)
            result = predict("win free money now", model_path=path)
        self.assertIsInstance(result["is_spam"], bool)
        self.assertIn(result["confidence"], {"high", "medium", "low"})
        self.assertTrue(result["top_features"], "expected top features from union pipeline")
        for feature in result["top_features"]:
            self.assertIn("word", feature)
            self.assertIn("weight", feature)
            self.assertIn(feature["direction"], {"spam", "ham"})

    def test_calibrated_union_pipeline(self):
        """CalibratedClassifierCV wrapper around the union pipeline stays explainable."""
        calibrated = CalibratedClassifierCV(FrozenEstimator(_union_model())).fit(
            [
                "free money now", "you won big", "claim your prize", "act now today",
                "win free cash", "congrats you won",
                "are you ok", "see you soon", "hello friend", "lunch tomorrow",
                "thanks for the call", "how was your day",
            ],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibrated_union.pkl"
            joblib.dump(calibrated, path)
            result = predict("win free money now", model_path=path)
        self.assertTrue(result["top_features"], "expected top features through wrapper+union")
        self.assertGreaterEqual(result["spam_probability"], 0.0)
        self.assertLessEqual(result["spam_probability"], 1.0)

    def test_calibrated_model(self):
        """CalibratedClassifierCV wrapper: real predict_proba + unwrapped top features."""
        from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

        base = Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", LinearSVC(max_iter=2000))]
        ).fit(
            ["win free money", "you won now", "how are you", "see you later", "claim prize today"],
            [1, 1, 0, 0, 1],
        )
        calibrated = CalibratedClassifierCV(FrozenEstimator(base)).fit(
            [
                "free money now", "you won big", "claim your prize", "act now today", "win free cash", "congrats you won",
                "are you ok", "see you soon", "hello friend", "lunch tomorrow", "thanks for the call", "how was your day",
            ],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibrated.pkl"
            joblib.dump(calibrated, path)
            result = predict("win free money now", model_path=path)
        self.assertTrue(result["top_features"], "expected top features through the wrapper")
        self.assertGreaterEqual(result["spam_probability"], 0.0)
        self.assertLessEqual(result["spam_probability"], 1.0)
        self.assertIn(result["confidence"], {"high", "medium", "low"})


class PredictInputGuardTest(unittest.TestCase):
    """Degenerate input must be reported, not turned into a confident answer."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = Path(self.tmp.name) / "model.pkl"
        joblib.dump(_tiny_model(), self.model_path)

    def test_rejects_non_string_message(self):
        for value in (None, 123, ["a"], {"a": 1}):
            with self.subTest(value=value):
                with self.assertRaises(TypeError):
                    predict(value, model_path=self.model_path)

    def test_empty_message_is_flagged_insufficient(self):
        result = predict("", model_path=self.model_path)
        self.assertTrue(result["insufficient_text"])
        self.assertFalse(result["is_spam"])
        self.assertEqual(result["confidence"], "low")

    def test_single_character_is_flagged_insufficient(self):
        """Regression: "a" used to come back as spam with medium confidence."""
        result = predict("a", model_path=self.model_path)
        self.assertTrue(result["insufficient_text"])
        self.assertFalse(result["is_spam"])

    def test_punctuation_only_is_insufficient(self):
        self.assertTrue(predict("!!!???...", model_path=self.model_path)["insufficient_text"])

    def test_digits_only_is_insufficient(self):
        self.assertTrue(predict("12345 67890", model_path=self.model_path)["insufficient_text"])

    def test_url_only_is_insufficient(self):
        """URLs are stripped, so a URL-only message has nothing left to judge."""
        self.assertTrue(predict("http://example.com", model_path=self.model_path)["insufficient_text"])

    def test_normal_message_is_not_insufficient(self):
        result = predict("win free money now", model_path=self.model_path)
        self.assertFalse(result["insufficient_text"])
        self.assertTrue(result["is_spam"])


class ModelCachingTest(unittest.TestCase):
    def test_load_model_returns_cached_instance(self):
        """Guards the perf fix: the API must not re-read the pickle per request."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            joblib.dump(_tiny_model(), path)
            self.assertIs(load_model(path), load_model(path))

    def test_missing_model_still_raises_system_exit(self):
        with self.assertRaises(SystemExit):
            load_model(Path("/nonexistent/model.pkl"))


if __name__ == "__main__":
    unittest.main()