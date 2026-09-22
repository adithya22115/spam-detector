"""Simple web API for the spam detection model."""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load configuration from the project's .env file (if python-dotenv is installed)
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Allow importing src modules without installing the package
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict  # noqa: E402

# Explicit template_folder: Flask(__name__) would resolve templates relative to
# however this module was imported, which breaks under gunicorn or when the
# module is loaded by path.
app = Flask(__name__, template_folder=str(PROJECT_ROOT / "app" / "templates"))

MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_PATH", "models/spam_model.pkl")

# Reject absurd payloads before they reach the vectorizer.
MAX_MESSAGE_CHARS = int(os.getenv("MAX_MESSAGE_CHARS", "20000"))


@app.route("/", methods=["GET"])
def index():
    """Serve the web UI."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """Classify a message. Always answers with JSON, including on errors."""
    # force=True tolerates a missing Content-Type header; silent=True turns a
    # malformed body into None so we can return our own JSON error instead of
    # Flask's HTML error page.
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    if "message" not in payload:
        return jsonify({"error": "Missing 'message' field"}), 400

    message = payload["message"]
    if not isinstance(message, str):
        return jsonify(
            {"error": f"'message' must be a string, got {type(message).__name__}"}
        ), 400
    if not message.strip():
        return jsonify({"error": "'message' must not be empty"}), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify(
            {"error": f"'message' must be at most {MAX_MESSAGE_CHARS} characters"}
        ), 400

    try:
        result = predict(message, model_path=MODEL_PATH)
    except (SystemExit, FileNotFoundError) as exc:
        return jsonify({"error": f"Model not available: {exc}"}), 500

    if result.get("insufficient_text"):
        return jsonify({"error": "'message' has no usable text content to classify"}), 400
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# Keep the API JSON-only: without these, Flask returns HTML error pages for
# 404/405 and for any unhandled exception.
@app.errorhandler(404)
def _not_found(_exc):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def _method_not_allowed(_exc):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def _internal_error(_exc):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1",
    )