"""Hyperparameter sweep for the spam detector.

Protocol (mirrors production train.py, test set stays untouched):
  1. Vectorize the train split once per feature config, reuse the matrix
     across classifier candidates, score each on the cal split.
  2. Pick the winner, calibrate it, evaluate once on test.
  3. Per-source error analysis on the test set.

Usage: python src/sweep.py
Writes results/sweep_summary.json.
"""

import json
import time
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from preprocessing import load_email_data, load_sms_data, preprocess_data

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_vectorizer(
    word_max_features=30_000,
    word_ngrams=(1, 2),
    char_max_features=30_000,
    char_ngrams=(3, 5),
    char_min_df=5,
) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    max_features=word_max_features,
                    ngram_range=word_ngrams,
                    sublinear_tf=True,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    max_features=char_max_features,
                    ngram_range=char_ngrams,
                    min_df=char_min_df,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def score(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_spam": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
    }


def make_clfs() -> list[tuple[str, object]]:
    return [
        ("LinearSVC C=1 (baseline)", LinearSVC(max_iter=2000)),
        ("LinearSVC C=0.5", LinearSVC(C=0.5, max_iter=2000)),
        ("LinearSVC C=2", LinearSVC(C=2, max_iter=2000)),
        ("LogisticRegression C=1", LogisticRegression(max_iter=2000)),
        ("ComplementNB a=0.3", ComplementNB(alpha=0.3)),
    ]


def main() -> None:
    sms = load_sms_data()
    email = load_email_data()
    sms["source"] = "sms"
    email["source"] = "email"
    df = pd.concat([sms, email], ignore_index=True)
    df = preprocess_data(df)
    print(f"Loaded {len(df):,} messages ({df['label'].value_counts().to_dict()})", flush=True)

    # Identical splits to src/train.py, plus a source column aligned by index
    X_tr, X_test, y_tr, y_test, src_tr, src_test = train_test_split(
        df["cleaned_message"],
        df["label"],
        df["source"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
    )
    X_fit = X_fit.reset_index(drop=True)
    X_cal = X_cal.reset_index(drop=True)
    y_fit = y_fit.reset_index(drop=True)
    y_cal = y_cal.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    src_test = src_test.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    print(f"fit={len(X_fit):,}  cal={len(X_cal):,}  test={len(X_test):,}", flush=True)

    rows = []

    def screen(name, clf, Xf, yf, Xc, t0):
        clf.fit(Xf, yf)
        pred = clf.predict(Xc)
        score_ = clf.decision_function(Xc) if hasattr(clf, "decision_function") else clf.predict_proba(Xc)[:, 1]
        m = score(y_cal, pred, score_)
        m |= {"candidate": name, "fit_seconds": round(time.time() - t0, 1)}
        rows.append(m)
        print(f"{name:34s} acc={m['accuracy']:.4f} f1={m['f1_spam']:.4f} "
              f"auc={m['roc_auc']:.4f} ({m['fit_seconds']}s)", flush=True)
        return m

    # Stage 1: default features, vectorize ONCE, reuse across classifiers
    print("\n-- Stage 1: classifier candidates on default features --", flush=True)
    t0 = time.time()
    vec = build_vectorizer()
    Xf_def = vec.fit_transform(X_fit, y_fit)
    Xc_def = vec.transform(X_cal)
    print(f"vectorized fit={Xf_def.shape} in {time.time() - t0:.1f}s", flush=True)
    fitted = []  # (name, vectorizer, clf) for winner reuse
    for name, clf in make_clfs():
        t0 = time.time()
        screen(name, clf, Xf_def, y_fit, Xc_def, t0)
        fitted.append((name, vec, clf))

    # Stage 2: feature-config variants (LinearSVC C=1 each)
    print("\n-- Stage 2: feature variants (LinearSVC C=1) --", flush=True)
    variants = [
        ("LinearSVC C=1 word 60k", dict(word_max_features=60_000)),
        ("LinearSVC C=1 word (1,3)", dict(word_ngrams=(1, 3))),
        ("LinearSVC C=1 char min_df=3", dict(char_min_df=3)),
    ]
    for name, kw in variants:
        t0 = time.time()
        v = build_vectorizer(**kw)
        Xf = v.fit_transform(X_fit, y_fit)
        Xc = v.transform(X_cal)
        clf = LinearSVC(max_iter=2000)
        screen(name, clf, Xf, y_fit, Xc, t0)
        fitted.append((name, v, clf))

    # Winner: reuse already-fitted vectorizer + classifier, calibrate, test once
    rows.sort(key=lambda r: (r["f1_spam"], r["accuracy"]), reverse=True)
    best_name = rows[0]["candidate"]
    best_vec, best_clf = next((v, c) for n, v, c in fitted if n == best_name)
    print(f"\nWinner on cal split: {best_name}", flush=True)

    pipe = Pipeline([("features", best_vec), ("clf", best_clf)])
    calibrated = CalibratedClassifierCV(FrozenEstimator(pipe))
    calibrated.fit(X_cal, y_cal)

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = score(y_test, y_pred, y_proba)
    test_metrics["brier"] = round(float(brier_score_loss(y_test, y_proba)), 4)
    print(f"\nTest (winner): {test_metrics}", flush=True)

    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix [ham,spam] rows=true:\n{cm}", flush=True)

    # Per-source error analysis
    results = pd.DataFrame(
        {"message": X_test, "label": y_test, "pred": y_pred, "source": src_test}
    )
    results["error"] = results["label"] != results["pred"]
    err_by_src = (
        results.groupby("source")["error"].agg(["sum", "count"]).assign(
            rate=lambda d: (d["sum"] / d["count"]).round(4)
        )
    )
    print("\nErrors by source:")
    print(err_by_src, flush=True)

    print("\nFalse positives (ham flagged as spam):")
    fps = results[(results["label"] == 0) & (results["pred"] == 1)]
    for _, r in fps.head(5).iterrows():
        print(f"  [{r['source']}] {r['message'][:90]}")
    print("\nFalse negatives (spam marked as ham):")
    fns = results[(results["label"] == 1) & (results["pred"] == 0)]
    for _, r in fns.head(5).iterrows():
        print(f"  [{r['source']}] {r['message'][:90]}")
    print(flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline_test_metrics": {
            "accuracy": 0.981, "f1_spam": 0.9791,
            "roc_auc": 0.9983, "brier": 0.0144,
        },
        "winner": best_name,
        "test_metrics": test_metrics,
        "cal_split_results": rows,
        "errors_by_source": err_by_src.to_dict(orient="index"),
    }
    with open(RESULTS_DIR / "sweep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Sweep summary saved to {RESULTS_DIR / 'sweep_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
