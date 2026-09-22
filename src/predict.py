"""Predict whether a message is spam using a trained model."""

import argparse
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib

from preprocessing import clean_text

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

TOP_FEATURES = 5

# Messages with fewer alphanumeric characters than this carry no usable signal.
# For such input the model falls back to its base rate, which is an artifact of
# an empty feature vector rather than a real prediction (the single character
# "a" used to come back as spam at p=0.70). We flag it instead of trusting it.
MIN_CONTENT_CHARS = 2

_CONTENT_RE = re.compile(r"\W")

# Lightweight content cues used to describe a message in plain language.
_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)
_MONEY_RE = re.compile(
    r"[$\u00a3\u20ac]\s?\d|\b\d+\s?(?:usd|dollars?|euros?|pounds?|inr|rs)\b",
    re.IGNORECASE,
)
_PRESSURE_WORDS = (
    "free", "claim", "prize", "win", "winner", "urgent", "immediately",
    "limited", "offer", "discount", "cash", "credit", "loan", "guaranteed",
    "verify", "click", "subscribe", "congratulations", "won", "now", "today",
)


def _join_phrases(items: list[str]) -> str:
    """Join phrases into a readable list: 'a, b and c'."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _message_signals(text: str, cleaned: str) -> list[str]:
    """Plain-language cues present in the message, for the description sentence."""
    signals: list[str] = []
    if _URL_RE.search(text):
        signals.append("contains a link")
    if _MONEY_RE.search(text):
        signals.append("mentions an amount of money")
    if text.count("!") >= 3:
        signals.append("uses repeated exclamation marks")
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.4:
        signals.append("is written mostly in capitals")
    words = set(cleaned.split())
    pressure = [w for w in _PRESSURE_WORDS if w in words]
    if pressure:
        quoted = [f"'{w}'" for w in pressure[:3]]
        signals.append(f"uses pressure wording such as {_join_phrases(quoted)}")
    return signals


def _describe_message(
    text: str,
    cleaned: str,
    is_spam: bool,
    spam_probability: float,
    confidence: str,
    top_features: list[dict],
) -> str:
    """A human-readable sentence describing what the model made of the message."""
    prob = spam_probability * 100
    signals = _message_signals(text, cleaned)

    if is_spam:
        started = f"Looks like spam ({prob:.1f}% spam, {confidence} confidence)."
        if signals:
            return (
                f"{started} The message {_join_phrases(signals)}, matching the kind of "
                "unsolicited, promotional or prize-style wording the model flags as unwanted."
            )
        return (
            f"{started} The wording matches patterns the model associates with "
            "unsolicited or promotional messages."
        )

    started = f"Looks like a normal message ({prob:.1f}% spam, {confidence} confidence)."
    if signals:
        return (
            f"{started} Although the message {_join_phrases(signals)}, the overall wording "
            "reads like ordinary conversation, so it is not flagged as spam."
        )
    return f"{started} The text reads like ordinary conversation with no obvious spam signals."


@lru_cache(maxsize=8)
def _load_model_cached(path_str: str, _mtime: float):
    """Load a pickled model once per (path, mtime).

    Cached so the Flask app does not re-read a 2.7 MB pickle on every request,
    and keyed on mtime so retraining is picked up without a process restart.
    """
    return joblib.load(path_str)


def load_model(path: Path = MODELS_DIR / "spam_model.pkl"):
    path = Path(path)
    if not path.exists():
        sys.exit(f"Model not found at {path}. Run `python src/train.py` first.")
    resolved = path.resolve()
    return _load_model_cached(str(resolved), resolved.stat().st_mtime)


def _content_length(cleaned_text: str) -> int:
    """Count the alphanumeric characters remaining after cleaning."""
    return len(_CONTENT_RE.sub("", cleaned_text))


def _confidence_label(spam_probability: float) -> str:
    """Map a spam probability to a confidence label based on distance from 0.5."""
    distance = abs(spam_probability - 0.5)
    if distance >= 0.3:
        return "high"
    if distance >= 0.1:
        return "medium"
    return "low"


def _fitted_transformers(feature_union) -> list:
    """Fitted transformers from a FeatureUnion, skipping dropped ones."""
    return [
        transformer
        for name, transformer in feature_union.transformer_list
        if transformer is not None and transformer != "drop"
    ]


def _feature_contributions(vectorizer, classifier, cleaned_text: str) -> list[tuple[str, float]]:
    """(feature name, tfidf value * coefficient) pairs for one message."""
    coef = classifier.coef_[0]
    contributions: list[tuple[str, float]] = []

    if hasattr(vectorizer, "transformer_list"):
        # FeatureUnion: coefficient blocks are stacked horizontally per transformer
        offset = 0
        for transformer in _fitted_transformers(vectorizer):
            sub = transformer.transform([cleaned_text])
            names = transformer.get_feature_names_out()
            for idx, value in zip(sub.indices, sub.data):
                weight = float(value * coef[offset + idx])
                if weight != 0:
                    contributions.append((names[idx], weight))
            offset += sub.shape[1]
        return contributions

    x = vectorizer.transform([cleaned_text])
    names = vectorizer.get_feature_names_out()
    for idx, value in zip(x.indices, x.data):
        weight = float(value * coef[idx])
        if weight != 0:
            contributions.append((names[idx], weight))
    return contributions


def _top_features(model, cleaned_text: str, n: int = TOP_FEATURES) -> list[dict]:
    """Words in the message that most influenced the prediction.

    Contribution = tfidf value * classifier coefficient, so positive weights
    push toward spam and negative weights toward ham. Supports both a single
    TF-IDF step ("tfidf") and a word+char FeatureUnion ("features"). Returns
    [] when the model does not expose coefficients (e.g. tree-based classifiers).
    """
    # Unwrap calibration wrappers (CalibratedClassifierCV / FrozenEstimator)
    while not hasattr(model, "named_steps"):
        for attr in ("estimator_", "estimator"):
            if hasattr(model, attr):
                model = getattr(model, attr)
                break
        else:
            return []
    try:
        vectorizer = model.named_steps["features"]
    except KeyError:
        vectorizer = model.named_steps.get("tfidf")
    classifier = model.named_steps.get("clf")
    if vectorizer is None or classifier is None or not hasattr(classifier, "coef_"):
        return []

    try:
        pairs = _feature_contributions(vectorizer, classifier, cleaned_text)
    except (AttributeError, IndexError):
        return []

    contributions = [
        {
            "word": name,
            "weight": round(weight, 4),
            "direction": "spam" if weight > 0 else "ham",
        }
        for name, weight in pairs
    ]
    contributions.sort(key=lambda c: abs(c["weight"]), reverse=True)
    return contributions[:n]


def _spam_probability(model, cleaned_text: str) -> float:
    """Estimate P(spam), falling back to sigmoid(decision_function) for
    classifiers without predict_proba (e.g. LinearSVC)."""
    try:
        return float(model.predict_proba([cleaned_text])[0][1])
    except AttributeError:
        score = float(model.decision_function([cleaned_text])[0])
        return 1.0 / (1.0 + math.exp(-score))


def predict(text: str, model_path: Path = MODELS_DIR / "spam_model.pkl") -> dict:
    """Classify one message.

    Returns is_spam, spam_probability, confidence, top_features, and
    insufficient_text. When insufficient_text is True the message had too
    little usable content to classify, so is_spam is forced to False and the
    confidence to "low" rather than reporting a meaningless prediction.
    """
    if not isinstance(text, str):
        raise TypeError(f"predict expects a str message, got {type(text).__name__}")
    model = load_model(model_path)
    cleaned = clean_text(text)
    insufficient = _content_length(cleaned) < MIN_CONTENT_CHARS
    label = model.predict([cleaned])[0]
    spam_probability = _spam_probability(model, cleaned)
    is_spam = bool(label) and not insufficient
    confidence = "low" if insufficient else _confidence_label(spam_probability)
    top_features = _top_features(model, cleaned)
    return {
        "text": text,
        "is_spam": is_spam,
        "spam_probability": spam_probability,
        "confidence": confidence,
        "insufficient_text": insufficient,
        "top_features": top_features,
        "description": _describe_message(
            text, cleaned, is_spam, spam_probability, confidence, top_features
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict spam for a message.")
    parser.add_argument("message", help="The message to classify.")
    parser.add_argument("--model", type=Path, default=None, help="Path to a trained model.")
    args = parser.parse_args()

    result = predict(args.message, args.model if args.model else MODELS_DIR / "spam_model.pkl")
    verdict = "SPAM" if result["is_spam"] else "HAM"
    print(f"{verdict} (spam probability: {result['spam_probability']:.2f}, "
          f"confidence: {result['confidence']})")
    print(result["description"])


if __name__ == "__main__":
    main()