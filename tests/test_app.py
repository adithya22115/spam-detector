"""Endpoint tests for the Flask API (app/app.py).

Covers the happy path plus malformed, missing, and wrongly-typed payloads.
Every error response must be JSON, not Flask's default HTML error page, because
these 500s were reachable before: a non-string 'message' (int/bool/list) went
straight into clean_text() and raised AttributeError.

Run with: python -m unittest discover tests
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _load_api_module():
    """Import app/app.py by path, so 'app' cannot resolve to the app/ directory."""
    spec = importlib.util.spec_from_file_location("spam_api", ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = _load_api_module()

# models/*.pkl is gitignored, so CI runs without a trained model. Tests that need
# real predictions skip there instead of failing; run src/train.py to enable them.
MODEL_AVAILABLE = (ROOT / "models" / "spam_model.pkl").exists()
requires_model = unittest.skipUnless(
    MODEL_AVAILABLE, "models/spam_model.pkl absent; run src/train.py first"
)


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        api.app.config.update(TESTING=True)
        cls.client = api.app.test_client()

    def assertJsonError(self, response, status: int):
        """Assert an error response is JSON with an 'error' key."""
        self.assertEqual(response.status_code, status)
        self.assertIn("application/json", response.headers.get("Content-Type", ""))
        body = response.get_json()
        self.assertIsInstance(body, dict)
        self.assertIn("error", body)
        return body


class HealthAndIndexTest(ApiTestCase):
    def test_health_returns_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_index_serves_html(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<html", response.data.lower())

    def test_unknown_route_returns_json_404(self):
        self.assertJsonError(self.client.get("/does-not-exist"), 404)


@requires_model
class PredictHappyPathTest(ApiTestCase):
    def test_classifies_spam(self):
        response = self.client.post(
            "/predict",
            json={"message": "Congratulations! You've won a free iPhone. Click here to claim."},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["is_spam"])
        self.assertGreater(body["spam_probability"], 0.5)
        self.assertIn(body["confidence"], {"high", "medium", "low"})
        self.assertIsInstance(body["top_features"], list)
        self.assertFalse(body["insufficient_text"])

    def test_classifies_ham(self):
        response = self.client.post(
            "/predict", json={"message": "Hi team, the meeting is at 4pm today."}
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["is_spam"])

    def test_accepts_json_without_content_type_header(self):
        """force=True keeps the endpoint tolerant of clients that omit the header."""
        response = self.client.post(
            "/predict",
            data=json.dumps({"message": "Free money now!"}),
            content_type="text/plain",
        )
        self.assertEqual(response.status_code, 200)

    def test_get_predict_returns_json_405(self):
        self.assertJsonError(self.client.get("/predict"), 405)


class PredictValidationTest(ApiTestCase):
    """Every one of these must be a 400 with a JSON body - none may 500."""

    def test_missing_message_field(self):
        body = self.assertJsonError(self.client.post("/predict", json={"msg": "wrong key"}), 400)
        self.assertEqual(body["error"], "Missing 'message' field")

    def test_empty_message(self):
        self.assertJsonError(self.client.post("/predict", json={"message": ""}), 400)

    def test_whitespace_only_message(self):
        self.assertJsonError(self.client.post("/predict", json={"message": "   \n\t "}), 400)

    def test_null_message(self):
        self.assertJsonError(self.client.post("/predict", json={"message": None}), 400)

    def test_integer_message(self):
        """Regression: this used to raise AttributeError and return a 500."""
        body = self.assertJsonError(self.client.post("/predict", json={"message": 12345}), 400)
        self.assertIn("must be a string", body["error"])

    def test_boolean_message(self):
        self.assertJsonError(self.client.post("/predict", json={"message": True}), 400)

    def test_list_message(self):
        self.assertJsonError(self.client.post("/predict", json={"message": [1, 2]}), 400)

    def test_object_message(self):
        self.assertJsonError(self.client.post("/predict", json={"message": {"a": 1}}), 400)

    def test_malformed_json_body(self):
        response = self.client.post(
            "/predict", data="{not valid json", content_type="application/json"
        )
        self.assertJsonError(response, 400)

    def test_empty_body(self):
        response = self.client.post("/predict", data="", content_type="application/json")
        self.assertJsonError(response, 400)

    def test_json_array_body(self):
        """A JSON array is valid JSON but not a valid payload object."""
        response = self.client.post(
            "/predict", data="[1,2,3]", content_type="application/json"
        )
        self.assertJsonError(response, 400)

    def test_message_exceeding_max_length(self):
        response = self.client.post("/predict", json={"message": "a" * (api.MAX_MESSAGE_CHARS + 1)})
        body = self.assertJsonError(response, 400)
        self.assertIn("at most", body["error"])

    @requires_model
    def test_message_with_no_usable_content(self):
        """Digits/punctuation only: nothing left to classify after cleaning."""
        for message in ("12345 67890", "!!!???...", "@@@ $$$"):
            with self.subTest(message=message):
                self.assertJsonError(self.client.post("/predict", json={"message": message}), 400)

    def test_all_errors_are_json_not_html(self):
        payloads = [
            ("{bad json", "application/json"),
            ("", "application/json"),
        ]
        for data, ctype in payloads:
            with self.subTest(data=data):
                response = self.client.post("/predict", data=data, content_type=ctype)
                self.assertIn("application/json", response.headers.get("Content-Type", ""))
                self.assertNotIn(b"<!doctype html", response.data.lower())


if __name__ == "__main__":
    unittest.main()
