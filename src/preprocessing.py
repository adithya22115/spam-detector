"""Text preprocessing utilities for SMS/email spam detection."""

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

SMS_DATA_FILE = "sms_spam.csv"
EMAIL_DATA_FILE = "email_spam.csv"

# Digits/symbols that spammers substitute for letters to dodge keyword filters.
# Applied only inside tokens that actually mix letters and leet characters, so
# "fr33" -> "free" while standalone numbers ("1800", "12345") are untouched.
LEETSPEAK_MAP = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
}

# Numbers are replaced with this token instead of being deleted, so numeric
# cues survive (a prize amount, a premium-rate number, an order id). One generic
# token lets the model learn "a number appeared here" without memorising values.
NUMBER_PLACEHOLDER = "num"

_TOKEN_RE = re.compile(r"[a-z0-9@$]+")
_HAS_LETTER_RE = re.compile(r"[a-z]")
_HAS_LEET_RE = re.compile(r"[0-9@$]")
_LEET_TRANS = str.maketrans(LEETSPEAK_MAP)

# NOTE: the [^\w\s] strip below also removes Unicode combining marks (category
# Mn), so Devanagari loses its vowel signs ("नमस्ते" -> "नमस त"). Preserving marks
# was implemented and measured: it made the model slightly WORSE on this dataset
# (accuracy 0.9813 -> 0.9809, SMS spam F1 0.850 -> 0.844) because the Enron
# mojibake then contributes noisy mark features. Deliberately left simple -
# NFKC below already covers the evasion trick that actually matters here
# (full-width look-alikes). See tests/test_preprocessing.py for the pinned
# behaviour.


def _decode_leet_token(token: str) -> str:
    """Expand leetspeak inside a single token, if it looks obfuscated.

    Only tokens that mix letters with leet characters are rewritten:
    "fr33" -> "free", "m0ney" -> "money", "cl1ck" -> "click". Pure numbers
    ("1800", "12345") and plain words are returned unchanged, so phone numbers
    and ordinary prose are unaffected.
    """
    if not _HAS_LETTER_RE.search(token) or not _HAS_LEET_RE.search(token):
        return token
    return token.translate(_LEET_TRANS)


def clean_text(text: str) -> str:
    """Normalize, lowercase, decode leetspeak, strip URLs/punctuation.

    NFKC normalization comes first so full-width and other compatibility
    look-alikes ("ＦＲＥＥ") fold onto their ASCII forms before tokenizing.
    Numbers are not discarded: each run of digits becomes NUMBER_PLACEHOLDER
    ("num"), so a prize amount or premium-rate number still leaves a feature.
    """
    if not isinstance(text, str):
        raise TypeError(f"clean_text expects a str, got {type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = _TOKEN_RE.sub(lambda match: _decode_leet_token(match.group(0)), text)  # leetspeak
    text = re.sub(r"\d+", f" {NUMBER_PLACEHOLDER} ", text)  # remaining numbers
    text = re.sub(r"[^\w\s]", " ", text)  # punctuation
    return re.sub(r"\s+", " ", text).strip()


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map string labels (ham/spam) to binary 0/1."""
    df = df.copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})
    return df


def load_sms_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load an SMS spam dataset with columns: label, message.

    Also accepts the original UCI v1/v2 tab-separated layout. Defaults to
    data/raw/sms_spam.csv, falling back to the legacy data/raw/spam.csv.
    """
    if path is None:
        path = RAW_DATA_DIR / SMS_DATA_FILE
        if not path.exists() and (RAW_DATA_DIR / "spam.csv").exists():
            path = RAW_DATA_DIR / "spam.csv"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No SMS dataset found at {path}. Download the UCI SMS Spam "
            f"Collection and save it as data/raw/{SMS_DATA_FILE}."
        )
    df = pd.read_csv(path, encoding="latin-1")
    if "label" not in df.columns and "v1" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]].copy()
    df["message"] = df["message"].fillna("").astype(str)
    return _normalize_labels(df)


def load_email_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load an email spam dataset.

    Accepts the Enron-style layout (Subject, Message, Spam/Ham) where the
    subject and body are combined into a single message, or a pre-normalized
    label/message layout. Defaults to data/raw/email_spam.csv.
    """
    path = Path(path) if path else RAW_DATA_DIR / EMAIL_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No email dataset found at {path}. Download an email spam "
            f"dataset and save it as data/raw/{EMAIL_DATA_FILE}."
        )
    df = pd.read_csv(path, encoding="utf-8")
    if "label" not in df.columns:
        if "Spam/Ham" in df.columns:
            df = df.rename(columns={"Spam/Ham": "label"})
        else:
            raise ValueError(
                "Email dataset must have a label column named 'label' or "
                f"'Spam/Ham'. Found columns: {list(df.columns)}"
            )
    if "message" not in df.columns:
        if "Message" not in df.columns:
            raise ValueError(
                "Email dataset must have a message column named 'message' or "
                f"'Message'. Found columns: {list(df.columns)}"
            )
        subject = df["Subject"].fillna("").astype(str) if "Subject" in df.columns else ""
        df["message"] = (subject + " " + df["Message"].fillna("").astype(str)).str.strip()
    df = df[["label", "message"]].copy()
    df["message"] = df["message"].fillna("").astype(str)
    return _normalize_labels(df)


def load_raw_data(path: Path | str | None = None, format: str = "auto") -> pd.DataFrame:
    """Load labeled spam data, auto-detecting SMS vs email format.

    format: "auto" (default, detects by filename), "sms", or "email".
    """
    if format not in {"auto", "sms", "email"}:
        raise ValueError("format must be one of: 'auto', 'sms', 'email'")
    if format == "sms":
        return load_sms_data(path)
    if format == "email":
        return load_email_data(path)
    name = Path(path).name.lower() if path else ""
    if "email" in name:
        return load_email_data(path)
    return load_sms_data(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean messages and return a processed DataFrame."""
    df = df.copy()
    df["cleaned_message"] = df["message"].fillna("").astype(str).apply(clean_text)
    return df


def save_processed_data(df: pd.DataFrame, filename: str = "processed_spam.csv") -> Path:
    """Save processed data to data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path