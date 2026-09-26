# Large-Scale Phishing URL Detection Through Structural and Webpage Feature Learning

Research prototype and interactive analysis application for classifying phishing and legitimate URLs with structural and webpage-derived features. The repository includes a tabular Random Forest model, a separate title-text NLP baseline, saved evaluation artifacts, and a Streamlit interface.

> **Research-use warning:** The bundled Random Forest evaluation reports perfect scores on a random held-out split. These results have not been independently reproduced or validated on a domain-, campaign-, source-, or time-separated dataset. They must not be interpreted as evidence of perfect real-world detection. See [Evaluation and interpretation](#evaluation-and-interpretation).

## Abstract

This project studies supervised phishing URL classification using the PhiUSIIL dataset snapshot bundled with the repository. The dataset contains 235,795 records and 50 numeric candidate predictors describing URL structure and webpage/HTML properties. A stratified 80/20 train/test split is used to fit and evaluate a 200-tree Random Forest. As a text-only comparison, a TF-IDF representation of webpage titles is evaluated with Logistic Regression on the same split strategy. An interactive Streamlit application exposes dataset exploration, model diagnostics, single-URL analysis, batch scoring, and title classification.

The saved Random Forest metrics are 1.0000 for accuracy, precision, recall, F1, and ROC-AUC. The title-only classifier reports 0.7036 accuracy and 0.6650 ROC-AUC. The difference motivates a careful examination of feature provenance, row-level split limitations, and possible dataset-specific signals. These numbers describe the bundled random split only; independent validation is needed before drawing claims about deployment performance.

**Keywords:** phishing detection, URL classification, webpage features, Random Forest, TF-IDF, Logistic Regression, cybersecurity, supervised learning.

## 1. Project goals and research questions

The implementation supports the following exploratory questions:

1. How well can the dataset's numeric URL and webpage features distinguish the provided phishing and legitimate labels under a stratified random split?
2. Which individual features and feature groups have the highest Random Forest impurity-based importance in this fitted model?
3. How much classification signal is available from webpage title text alone using TF-IDF and Logistic Regression?
4. How can dataset distributions and model outputs be inspected interactively?

These are repository-level research questions, not claims that the current evaluation establishes real-world generalization or causality.

## 2. Dataset

The app reads `Datasets/PhiUSIIL_Phishing_URL_Dataset.csv`.

| Property | Value |
| --- | ---: |
| Records | 235,795 |
| Columns | 56 |
| Numeric model predictors | 50 |
| Phishing records (`label = 0`) | 100,945 (42.81%) |
| Legitimate records (`label = 1`) | 134,850 (57.19%) |
| Missing cells in bundled CSV snapshot | 0 |
| Exact duplicate rows in bundled CSV snapshot | 0 |

The 50 predictors are numeric. `FILENAME`, `URL`, `Domain`, `TLD`, and `Title` are excluded from the tabular model; `label` is the target. The NLP model uses the `Title` field and the same `label` target. Class encoding is consistent throughout the code: `0 = phishing`, `1 = legitimate`.

### 2.1 Predictor inventory

The feature order is recorded in [`models/feature_columns.json`](models/feature_columns.json) and is the required order for batch scoring.

**URL and structural features (22):**

`URLLength`, `DomainLength`, `IsDomainIP`, `URLSimilarityIndex`, `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`.

**Webpage and HTML features (28):**

`LineOfCode`, `LargestLineLength`, `HasTitle`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`.

Feature names preserve the spelling in the supplied PhiUSIIL schema, including `NoOfDegitsInURL`, `DegitRatioInURL`, and `SpacialCharRatioInURL`.

### 2.2 Dataset provenance

The repository contains the dataset CSV but does not include a canonical publication citation, DOI, download URL, or license statement. Before using the data in a paper or redistributing it, identify and cite the original dataset publication or official distribution page and verify its terms. Do not infer a citation or license from the filename alone.

## 3. Methods

### 3.1 Tabular Random Forest

The implementation in `src/train_model.py`:

1. Reads the full CSV and selects all columns except `FILENAME`, `URL`, `Domain`, `TLD`, `Title`, and `label` as predictors.
2. Creates an 80/20 train/test split with `random_state=42` and stratification by label.
3. Fits `sklearn.ensemble.RandomForestClassifier` with 200 estimators, unlimited tree depth, `class_weight="balanced_subsample"`, `n_jobs=-1`, and `random_state=42`.
4. Calculates accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, precision-recall curve, and impurity-based feature importances on the held-out rows.

The saved run contains 188,636 training rows and 47,159 test rows. This is a row-wise random split; the code does not group rows by registrable domain, campaign, source, or time.

### 3.2 Title-only NLP baseline

The separate implementation in `src/train_nlp_model.py`:

1. Converts `Title` values to strings and uses the same 80/20 stratified split settings (`random_state=42`).
2. Tokenizes contiguous alphabetic sequences of at least two characters and lowercases them.
3. Fits `TfidfVectorizer` with English stop words, up to 5,000 features, unigrams and bigrams, and `min_df=3`.
4. Fits `LogisticRegression` with `max_iter=1000` and `class_weight="balanced"`.
5. Saves test metrics, a confusion matrix, the largest-magnitude predictive terms, class-specific word counts, and title-length samples.

This classifier is a **separate title-only baseline**. The single-URL Random Forest workflow does not combine its probability with the NLP classifier; title analysis is available separately under **NLP Insights**.

### 3.3 Metrics and class convention

The training scripts call scikit-learn's binary precision, recall, F1, and ROC-AUC functions with their default positive class, `label = 1` (legitimate). Therefore, the precision/recall/F1 columns below are for the legitimate class. The confusion matrices use label order `[0, 1]`: rows are actual phishing and actual legitimate; columns are predicted phishing and predicted legitimate.

## 4. Evaluation and interpretation

The values below are read from the model artifacts currently committed in `models/`.

| Model | Accuracy | Precision (legit) | Recall (legit) | F1 (legit) | ROC-AUC (legit) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest, 50 numeric features | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| TF-IDF + Logistic Regression, title only | 0.7036 | 0.6626 | 0.9816 | 0.7911 | 0.6650 |

**Random Forest confusion matrix:** `[[20189, 0], [0, 26970]]`.

**Title NLP confusion matrix:** `[[6708, 13481], [497, 26473]]`.

The title model correctly identifies 26,473 of 26,970 legitimate test rows, but only 6,708 of 20,189 phishing rows are predicted as phishing at the default threshold. Its high recall value in the table refers to the positive class (legitimate), not phishing. The ROC-AUC is included because accuracy alone can obscure class-specific behavior.

### 4.1 Important validity caveat

The perfect Random Forest result is unusual for an operational phishing detector and should be treated as a diagnostic finding, not a deployment claim. The repository's evaluation is limited to a random stratified row split. It does not establish performance on unseen domains, new campaigns, later time periods, different data sources, or production traffic. Exact duplicate rows are absent in the bundled CSV, but that alone does not establish independence between rows or domains.

Before publication or operational use, investigate:

- Whether near-duplicate URLs, domains, or campaign templates cross the train/test boundary.
- The provenance and construction of high-importance fields, especially `URLSimilarityIndex`, `NoOfExternalRef`, `LineOfCode`, and webpage counts.
- Potential target-derived or collection-source signals in the predictors.
- Performance under grouped-by-domain and temporal holdouts, and on an independently collected dataset.
- Per-class precision, recall, F1, calibration, confidence intervals, and threshold selection based on the intended cost of false positives and false negatives.

The auxiliary lookup tables used for arbitrary-URL feature approximation are built from the full CSV after the estimator split. They are not used to calculate the stored held-out estimator metrics, but this means the complete inference pipeline has not been evaluated with every reference artifact isolated to training data. A rigorous follow-up should build all reference artifacts using training-fold data only.

## 5. Application

The Streamlit application is launched from `app.py`. The left navigation contains the following views:

- **Overview:** one scrollable home page containing dataset analytics, descriptive statistics, feature comparisons, feature importances, correlation analysis, model metrics, and model diagnostic plots. Its exploratory distribution charts use a reproducible sample of up to 20,000 rows (`random_state=42`).
- **Check a URL:** computes lexical features from a submitted URL and predicts with the tabular Random Forest. Live webpage retrieval is optional. When retrieval is disabled or fails, webpage columns are populated from legitimate-class median defaults and the app warns that this is an approximation.
- **Batch CSV Prediction:** scores a CSV containing all 50 model predictor columns. Extra columns are retained in the output. The downloadable CSV adds `prediction` and `legitimate_probability`; the latter is the probability of class 1 (legitimate).
- **Dataset Explorer:** interactive class distribution, feature histogram and box plot, scatter plot, top-feature correlation heatmap, and a raw sample. It displays up to 20,000 rows; the scatter plot is further sampled to at most 3,000 rows.
- **Model Performance:** saved Random Forest metrics, confusion matrix, top feature importances, ROC curve, and precision-recall curve.
- **NLP Insights:** the separate title-only classifier, title input, evaluation diagnostics, title-length plots, predictive terms, and word frequency visualizations.

### 5.1 Single-URL feature generation

`src/feature_extraction.py` normalizes a URL by adding `http://` when no scheme is present, then computes URL-only values such as length, digit/letter ratios, subdomain count, HTTPS status, and percent-encoding counts. It also estimates `URLSimilarityIndex` by comparing the host with stored legitimate domains, estimates TLD legitimacy from a stored lookup, and estimates character probability from stored URL character frequencies. These estimates are approximations and may not reproduce the source dataset's original feature-generation process.

For live analysis, the app parses fetched HTML for title, forms, links, scripts, images, metadata, redirects, and selected page text. This is static HTML analysis; it does not execute JavaScript or render a browser DOM. JavaScript-heavy pages can therefore be represented incompletely.

### 5.2 Live-fetch security and privacy

Live fetch is opt-in. The code accepts only HTTP and HTTPS, resolves and rejects an initial host when an address is classified as private, loopback, link-local, multicast, reserved, or unspecified, sets a six-second request timeout, limits redirects to five, and reads at most 3 MB of the primary HTML response.

**This is not a hardened browser or a complete SSRF boundary.** Redirect destinations are not revalidated before requests follow them, and the separate `robots.txt` request does not use the same streamed response-size limit. Do not expose this feature as a public fetch service without additional network-level egress controls and redirect-by-redirect address validation. Fetching a URL also contacts that external host and may disclose the requester's IP address and user-agent to it.

## 6. Installation and setup

### Requirements

- Python 3.10 or newer is recommended (the source uses modern Python typing syntax).
- Dependencies are listed in `requirements.txt`: pandas, NumPy, scikit-learn, Streamlit, joblib, Plotly, requests, and Beautiful Soup.
- The bundled dataset and model artifacts are required for the normal app workflow.

### Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Run the application

From the repository root:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, typically `http://localhost:8501`.

### Retrain the models

Train the structural/webpage Random Forest and regenerate its feature and evaluation artifacts:

```bash
python -m src.train_model
```

Train the independent title NLP classifier and its artifacts:

```bash
python -m src.train_nlp_model
```

Training reads the CSV from `Datasets/` and writes generated files to `models/`. The two commands are independent; running one does not retrain the other. The full dataset is loaded into memory, and Random Forest fitting may require substantial RAM and CPU time. Dependency versions are not pinned, so exact bit-for-bit reproduction across environments is not guaranteed.

## 7. Batch prediction input schema

The upload must include every predictor in `models/feature_columns.json`. Column order in the uploaded CSV is not important because the app selects the stored feature order, but column names must match exactly and values must be usable by the fitted estimator. The `label` column is not required. A sample CSV schema can be obtained by selecting the 50 predictor columns from the bundled dataset.

The app checks for missing columns but does not provide comprehensive data validation for malformed, missing, nonnumeric, or out-of-distribution values. Validate batch inputs before scoring. The `legitimate_probability` output is `predict_proba(... )[:, 1]`, i.e. class 1 probability; `prediction` contains `legitimate` or `phishing`.

## 8. Repository structure

```text
.
|-- app.py
|-- README.md
|-- requirements.txt
|-- Datasets/
|   `-- PhiUSIIL_Phishing_URL_Dataset.csv
|-- models/
|   |-- model.joblib
|   |-- feature_columns.json
|   |-- feature_defaults.json
|   |-- tld_lookup.json
|   |-- legit_domains.json
|   |-- char_freq.json
|   |-- metrics.json
|   |-- correlation.json
|   |-- nlp_model.joblib
|   |-- nlp_vectorizer.joblib
|   |-- nlp_metrics.json
|   |-- word_frequencies.json
|   `-- title_lengths.json
`-- src/
	|-- feature_extraction.py
	|-- train_model.py
	`-- train_nlp_model.py
```

Model files use joblib serialization. Load only artifacts from a trusted source; Python pickle-based formats can execute code during deserialization.

## 9. Limitations and responsible use

- Dataset labels, collection process, time period, sampling design, and canonical citation are not documented in this repository. The results inherit those limitations.
- URL-derived lookup approximations, simple TLD parsing, and heuristic HTML extraction may differ from the dataset's feature-generation methods.
- The live HTML parser does not render scripts, follow client-side navigation, or inspect page behavior in a browser sandbox.
- A probability from a classifier is not necessarily calibrated, and the default decision threshold is not selected from an operational cost analysis.
- No independent external validation, temporal evaluation, domain-grouped evaluation, calibration study, or confidence interval is bundled.
- The current random-split Random Forest score is especially important to re-examine for data leakage and source or domain overlap.
- No dedicated automated test suite is included in this repository.

This software is a research and educational prototype. It should support, not replace, security review and threat-intelligence workflows. Do not use the prediction as the sole basis for blocking a site, accusing an operator, or making a high-impact decision. Review false positives, false negatives, privacy implications, and applicable dataset terms before deployment.

## 10. Reproducibility checklist

For a stronger experimental report, record the dataset's canonical citation and checksum, Python and dependency versions, model artifact hashes, random seeds, and feature-generation version. Then:

1. Rebuild reference lookups from training data only.
2. Split by registrable domain, campaign, collection source, and/or time where appropriate.
3. Check for exact and near duplicates across partitions.
4. Report per-class metrics and the confusion matrix, with the positive class stated explicitly.
5. Evaluate calibration and threshold-dependent costs on a separate validation set.
6. Test on a genuinely external dataset collected after the training data.

The saved outputs in `models/metrics.json` and `models/nlp_metrics.json` document the currently bundled evaluation run; retraining can change those outputs when library versions, source data, or implementation details change.

## Citation and attribution

The repository does not currently include author names, institutional affiliations, a publication date, or the original PhiUSIIL dataset citation. Add the verified dataset citation and project authorship metadata here before treating this README as a formal paper supplement. Cite the original data source separately from this software repository, and confirm redistribution permissions for the CSV and model artifacts.
