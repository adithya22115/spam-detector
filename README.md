# SMS / Email Spam Detection

Detect spam in SMS and email messages using machine learning: word + char
TF-IDF features feeding a probability-calibrated Linear SVM.

## Project Structure

```
SMS-EMAIL-SPAM-DETECTION
├── data
│   ├── raw          # Datasets (sms_spam.csv, email_spam.csv)
│   └── processed    # Cleaned/preprocessed datasets
├── notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_transformer_model.ipynb   # distilBERT fine-tuning (optional deps)
├── models           # Trained model artifacts (.pkl, transformer/)
├── src
│   ├── preprocessing.py  # Text cleaning and data loading (SMS + email)
│   ├── train.py          # Train, evaluate and export the model
│   ├── sweep.py          # Hyperparameter sweep (writes results/sweep_summary.json)
│   └── predict.py        # Classify new messages
├── app
│   ├── app.py            # Flask API
│   └── templates/        # Web UI
├── tests                 # Unit + API + model-quality tests
├── tools
│   └── verify_model.py   # Re-check the exported model end to end
├── results          # Evaluation metrics + confusion matrices
├── .github/workflows/ci.yml       # Tests on push/PR
├── .env.example     # Optional API configuration
├── requirements.txt
├── requirements-transformers.txt  # Optional: for notebook 04
├── Dockerfile       # Containerize the API
└── README.md
```

## Datasets

| File                   | Source                                                                                          | Rows   |
| ---------------------- | ----------------------------------------------------------------------------------------------- | ------ |
| `data/raw/sms_spam.csv`   | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)        | 5,572  |
| `data/raw/email_spam.csv` | [Enron-Spam dataset (preprocessed CSV)](https://github.com/MWiechmann/enron_spam_data)         | 33,716 |

Both files are normalized to `label,message` columns. The loaders also accept the
original layouts: UCI `v1`/`v2` tab-separated files, or Enron-style
`Subject, Message, Spam/Ham` CSVs (subject and body are combined automatically).

## Setup

```bash
python -m venv .venv
# Windows (Git Bash / PowerShell)
.venv/Scripts/activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

To skip the venv, `pip install -r requirements.txt` with your system Python also works.

## Usage

1. **Train** (defaults to both datasets combined):
   ```bash
   python src/train.py                     # train on SMS + email
   python src/train.py --dataset sms       # train on SMS only
   python src/train.py --dataset email     # train on email only
   python src/train.py --data path/to.csv  # train on a custom CSV
   ```
2. **Predict:**
   ```bash
   python src/predict.py "Congratulations! You've won a free iPhone. Click here to claim."
   ```
3. **Serve via API:**
   ```bash
   python app/app.py
   curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" \
        -d '{"message": "Free money now!"}'
   ```
4. **Test / verify:**
   ```bash
   python -m unittest discover tests        # unit, API and model-quality tests
   python tools/verify_model.py             # reproduce metrics + adversarial battery
   ```

## Text preprocessing

`clean_text()` in `src/preprocessing.py` is applied at **both** train and inference
time. Order matters:

1. **NFKC normalize** — folds full-width look-alikes onto ASCII, so `ＦＲＥＥ` → `FREE`.
2. **Lowercase.**
3. **Strip URLs.**
4. **Decode leetspeak inside mixed tokens** — `fr33` → `free`, `cl1ck` → `click`,
   `m0ney` → `money`. Only tokens mixing letters *and* leet characters are rewritten,
   so standalone numbers (`1800`, `12345`) are left alone and later removed.
5. **Strip remaining digits, then punctuation, then collapse whitespace.**

Step 4 is why obfuscated spam is caught. Digits used to be deleted outright, which
*destroyed* the obfuscation instead of normalising it (`fr33` became `fr`), and
leetspeak spam slipped through.

## API

| Method | Path       | Description                    |
| ------ | ---------- | ------------------------------ |
| GET    | `/`        | Web UI                         |
| GET    | `/health`  | `{"status": "ok"}`             |
| POST   | `/predict` | Classify one message           |

`POST /predict` takes `{"message": "..."}` and returns:

```json
{
  "text": "Free money now!",
  "is_spam": true,
  "spam_probability": 0.98,
  "confidence": "high",
  "insufficient_text": false,
  "top_features": [{"word": "free", "weight": 1.23, "direction": "spam"}]
}
```

Every error is JSON (never an HTML error page):

| Condition                                        | Status |
| ------------------------------------------------ | ------ |
| Body is not a JSON object / malformed JSON        | 400    |
| `message` missing, empty, or whitespace-only      | 400    |
| `message` is not a string (e.g. `12345`, `[]`)    | 400    |
| `message` longer than `MAX_MESSAGE_CHARS`         | 400    |
| `message` has no usable text after cleaning       | 400    |
| Model file missing                                | 500    |

Configuration is read from the environment (see `.env.example`): `FLASK_HOST`,
`FLASK_PORT`, `FLASK_DEBUG`, `MODEL_PATH`, `MAX_MESSAGE_CHARS`.

### Short input

Messages with fewer than `MIN_CONTENT_CHARS` alphanumeric characters after cleaning
(`""`, `"a"`, `"!!!"`, `"12345"`) carry no signal — for those the vectorizer produces
an empty feature row and the model returns its base rate. Rather than dressing that
up as a prediction, `predict()` sets `insufficient_text: true`, forces
`is_spam: false`, and reports `confidence: "low"`; the API returns 400.

## Transformer model (notebook 04)

Fine-tune distilBERT and compare it with the Linear SVM baseline:

```bash
pip install -r requirements-transformers.txt   # CPU-only torch: see file header
```

Then run `notebooks/04_transformer_model.ipynb`. Results land in `results/`.

## Docker

```bash
docker build -t spam-detector .
docker run -p 5000:5000 spam-detector
```

The image bundles the trained model, so it serves `/predict` out of the box.

## Results

`results/` holds the evaluation artifacts (`metrics_*.json`,
`confusion_matrix_*.png`, `classification_report_*.txt`). The held-out test set is
7,858 messages (20%, `random_state=42`, stratified) that the model never saw.

**Overall — calibrated Linear SVM (word + char TF-IDF):**

| Metric    | Value  |
| --------- | ------ |
| accuracy  | 0.9813 |
| spam F1   | 0.9794 |
| ROC-AUC   | 0.9983 |
| Brier     | 0.0144 |

**Per source — read this before quoting the 98%:**

| Source | n     | Accuracy | Spam F1 |
| ------ | ----- | -------- | ------- |
| email  | 6,762 | 0.9846   | 0.9848  |
| sms    | 1,096 | 0.9608   | 0.8502  |

The aggregate is dominated by email (86% of the test set). **SMS spam detection is
~13 F1 points weaker** than email, so the headline number overstates SMS
performance. Both are reported here deliberately.

**Model comparison:**

| Model                                | Accuracy | Spam F1 | ROC-AUC | Brier  |
| ------------------------------------ | -------- | ------- | ------- | ------ |
| Calibrated Linear SVM (word + char)  | 0.9813   | 0.9794  | 0.9983  | 0.0144 |
| Linear SVM (word-only, pre-tuning)   | 0.9706   | 0.9676  | 0.9962  | 0.0225 |
| distilBERT (notebook 04, 1 epoch)    | 0.9500   | 0.9455  | 0.9801  | 0.0442 |

Probability calibration matters here: the raw word-only SVM had a Brier of 0.0225,
and `CalibratedClassifierCV` + `FrozenEstimator` brings it to 0.0144, which is what
makes the `confidence` labels in `src/predict.py` meaningful.

### Reproducing these numbers

```bash
python tools/verify_model.py
```

It rebuilds the exact `train.py` split, scores the **exported** pickle on it, diffs
the result against `results/metrics_all.json`, and runs an adversarial battery
(leetspeak, full-width look-alikes, HTML, oversized, empty). Training is seeded
(`random_state=42`), so retraining reproduces the table above exactly.

`results/baseline/` and `models/spam_model_baseline.pkl` keep the pre-leetspeak
model for comparison.

## Limitations

Known and deliberate, rather than undiscovered:

- **SMS is much weaker than email** (F1 0.85 vs 0.98). Email supplies 86% of the
  training rows, so SMS spam patterns are under-represented.
- **Numeric cues are discarded.** All digits are stripped, so a message whose only
  spam signal is a prize amount (`"C0NGRATULATIONS! U h4ve w0n $1000000"` → cleaned
  to `congratulations u have won`) can be missed. It is the one case the verification
  battery still fails.
- **English-only.** Non-Latin scripts are not supported, and combining marks are
  dropped (`नमस्ते` → `नमस त`) because punctuation stripping removes category-Mn
  characters. Preserving them was measured and made the model slightly worse
  (accuracy 0.9813 → 0.9809, SMS F1 0.850 → 0.844) as Enron mojibake then contributes
  noisy mark features.
- **Homoglyph evasion beyond NFKC is not covered** — `ＦＲＥＥ` is handled, a Cyrillic
  `а` standing in for a Latin `a` is not.
- **Very short messages are refused, not classified** (see "Short input" above).
- **The Flask app uses the development server**; put it behind a real WSGI server
  for anything public.

## Roadmap

- [x] Download real SMS and email spam datasets (`data/raw`)
- [x] Train a baseline model end-to-end
- [x] Explore the data in notebook 01
- [x] Preprocess and vectorize text in notebook 02
- [x] Compare models in notebook 03 (Linear SVM wins)
- [x] Calibrate probabilities (Brier 0.059 → 0.022)
- [x] Fine-tune distilBERT in notebook 04 and compare against the baseline
- [x] Tune the baseline with word + char TF-IDF features
- [x] Containerize the API with Docker
- [x] Normalize leetspeak instead of deleting digits (catches `fr33`, `cl1ck`)
- [x] NFKC-normalize full-width look-alike evasion
- [x] Report per-source metrics instead of a single email-dominated average
- [x] Guard degenerate/short input instead of predicting from an empty vector
- [x] API tests + model-quality regression tests, run in CI
- [ ] Improve SMS recall (class weighting or source-aware sampling)
- [ ] Replace digit stripping with a numeric placeholder to keep prize-amount cues
