"""Train a spam detection model and save it to models/.

Production configuration: calibrated Linear SVM over word + char TF-IDF
features, tuned in notebooks/03 (see README "Results").
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless: save plots to files instead of showing them
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from preprocessing import (
    load_email_data,
    load_raw_data,
    load_sms_data,
    preprocess_data,
    save_processed_data,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _load_datasets(dataset: str, data_path: Path | None) -> pd.DataFrame:
    """Load the requested dataset(s) and return a label/message/source DataFrame.

    The `source` column is carried through to the saved metrics so the
    per-dataset breakdown (SMS vs email) is reproducible from results/.
    """
    if data_path is not None:
        df = load_raw_data(data_path)
        df["source"] = "custom"
    elif dataset == "sms":
        df = load_sms_data()
        df["source"] = "sms"
    elif dataset == "email":
        df = load_email_data()
        df["source"] = "email"
    else:  # all
        sms = load_sms_data()
        email = load_email_data()
        sms["source"] = "sms"
        email["source"] = "email"
        df = pd.concat([sms, email], ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a spam detection model.")
    parser.add_argument(
        "--dataset",
        choices=["sms", "email", "all"],
        default="all",
        help="Which dataset(s) to train on (default: all).",
    )
    parser.add_argument("--data", type=Path, default=None, help="Path to a single custom CSV.")
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / "spam_model.pkl",
        help="Where to save the trained model (default: models/spam_model.pkl).",
    )
    args = parser.parse_args()

    df = _load_datasets(args.dataset, args.data)
    print(f"Loaded {len(df):,} messages "
          f"({df['label'].value_counts().to_dict()})")

    df = preprocess_data(df)
    save_processed_data(df, f"processed_{args.dataset}.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    # Hold out 20% of the training set for probability calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Word n-grams catch spam vocabulary; char n-grams catch obfuscated spam
    # (e.g. "fr33", "cl1ck"). This combination won the tuning sweep in
    # notebooks/03 over word-only TF-IDF (+1.1pp accuracy, +1.1pp spam F1).
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                max_features=30_000, ngram_range=(1, 2), sublinear_tf=True
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                max_features=30_000,
                                ngram_range=(3, 5),
                                min_df=5,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            ("clf", LinearSVC(max_iter=2000)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Calibrate on the held-out split so predict_proba reflects true accuracy,
    # keeping the confidence labels in src/predict.py trustworthy.
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline))
    calibrated.fit(X_cal, y_cal)

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]

    print("\nTest set evaluation:")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Per-source breakdown: the headline accuracy is dominated by email, so the
    # SMS numbers are reported separately rather than averaged away.
    source_test = df.loc[X_test.index, "source"]
    per_source: dict[str, dict] = {}
    print("Per-source evaluation:")
    for src in sorted(source_test.unique()):
        mask = (source_test == src).to_numpy()
        per_source[src] = {
            "accuracy": round(float(accuracy_score(y_test[mask], y_pred[mask])), 4),
            "f1_spam": round(float(f1_score(y_test[mask], y_pred[mask])), 4),
            "support": int(mask.sum()),
        }
        print(
            f"  {src:<7} n={mask.sum():>5}  acc={per_source[src]['accuracy']:.4f}  "
            f"f1_spam={per_source[src]['f1_spam']:.4f}"
        )

    results_dir = MODELS_DIR.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "calibrated Linear SVM (word + char TF-IDF)",
        "dataset": args.dataset,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_spam": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "brier": round(float(brier_score_loss(y_test, y_proba)), 4),
        "per_source": per_source,
    }
    with open(results_dir / f"metrics_{args.dataset}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["ham", "spam"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix — {args.dataset}")
    fig.tight_layout()
    fig.savefig(results_dir / f"confusion_matrix_{args.dataset}.png", dpi=150)
    plt.close(fig)

    (results_dir / f"classification_report_{args.dataset}.txt").write_text(
        classification_report(y_test, y_pred, target_names=["ham", "spam"])
    )
    print(f"Results saved to {results_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, args.out)
    print(f"Calibrated model saved to {args.out}")


if __name__ == "__main__":
    main()