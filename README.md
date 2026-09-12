# Large-Scale-Phishing-URL-Detection-Through-Structural-and-Webpage-Feature-Learning
This study investigates large-scale phishing URL detection using structural characteristics of URLs and webpage-based features to distinguish legitimate websites from phishing websites. Using the PhiUSIIL phishing URL dataset containing 235,795 URLs, the study analyzes URL and webpage characteristics and ML models for phishing classification.

## Streamlit App

An interactive Streamlit app is included to explore the dataset and run the trained model.

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train the model

Trains a Random Forest classifier on all structural + webpage features and saves artifacts to `models/`:

```bash
python -m src.train_model
```

### Run the app

```bash
streamlit run app.py
```

The app has five sections:
- **Overview** — dataset summary and headline model metrics.
- **Check a URL** — classify a single URL. Lexical/structural features (URL length, digit/letter ratios, TLD stats, obfuscation, etc.) are computed directly from the URL string. Optionally fetch the live page (with SSRF protections: blocks private/internal addresses, caps redirects and response size) to also extract webpage features (title, forms, scripts, images, social links, etc.); otherwise these fall back to dataset median values, which is clearly flagged as an approximation.
- **Batch CSV Prediction** — upload a CSV matching the dataset's feature schema for bulk scoring, with a downloadable results file.
- **Dataset Explorer** — interactive class distribution and feature histograms.
- **Model Performance** — accuracy/precision/recall/F1/ROC-AUC, confusion matrix, and feature importances.

### Notes & limitations
- The model is trained on all 50 numeric structural/webpage features from the dataset (raw identifiers like `FILENAME`, `URL`, `Domain`, `TLD`, `Title` are excluded).
- `URLSimilarityIndex`, `TLDLegitimateProb`, `URLCharProb`, and `CharContinuationRate` are approximated for arbitrary new URLs using lookup tables built from the training data; they may not exactly reproduce the original dataset's methodology.
- Live webpage feature extraction uses heuristics (e.g., regex/keyword search, tag counts) and may behave less accurately on modern JavaScript-heavy single-page sites that render most content client-side.
