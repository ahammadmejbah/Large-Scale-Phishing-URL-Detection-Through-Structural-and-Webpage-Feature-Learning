"""Train the phishing URL detection model on the PhiUSIIL dataset.

Produces the following artifacts under `models/`:
  - model.joblib            trained RandomForestClassifier
  - feature_columns.json    ordered list of feature columns used by the model
  - tld_lookup.json         TLD -> TLDLegitimateProb lookup table
  - legit_domains.json      sample of known-legitimate domains (for URL similarity approximation)
  - char_freq.json          character frequency table (for URLCharProb approximation)
  - metrics.json            evaluation metrics + feature importances
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "Datasets" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"

# Columns that identify a row or hold free text/high-cardinality strings and are
# therefore excluded from the numeric feature set used to train the model.
NON_FEATURE_COLUMNS = ["FILENAME", "URL", "Domain", "TLD", "Title", "label"]

MAX_LEGIT_DOMAINS = 40000  # cap stored domain list to keep artifact size reasonable


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def train_and_save() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df = load_dataset()

    feature_columns = build_feature_columns(df)
    X = df[feature_columns]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Downsample curve points so the JSON artifact stays small.
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    curve_step = max(len(fpr) // 200, 1)
    pr_step = max(len(precision_curve) // 200, 1)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_meaning": {"0": "phishing", "1": "legitimate"},
        "roc_curve": {"fpr": fpr[::curve_step].tolist(), "tpr": tpr[::curve_step].tolist()},
        "pr_curve": {
            "precision": precision_curve[::pr_step].tolist(),
            "recall": recall_curve[::pr_step].tolist(),
        },
        "feature_importances": dict(
            sorted(
                zip(feature_columns, model.feature_importances_.tolist()),
                key=lambda kv: kv[1],
                reverse=True,
            )
        ),
    }

    # --- Auxiliary lookup artifacts for approximating features from a raw URL ---
    tld_lookup = (
        df.groupby("TLD")["TLDLegitimateProb"].mean().round(6).to_dict()
    )

    legit_domains = df.loc[df["label"] == 1, "Domain"].dropna().unique().tolist()
    if len(legit_domains) > MAX_LEGIT_DOMAINS:
        rng = np.random.default_rng(42)
        legit_domains = rng.choice(
            legit_domains, size=MAX_LEGIT_DOMAINS, replace=False
        ).tolist()

    char_counts: dict[str, int] = {}
    total_chars = 0
    for url in df["URL"].astype(str):
        for ch in url:
            char_counts[ch] = char_counts.get(ch, 0) + 1
            total_chars += 1
    char_freq = {ch: count / total_chars for ch, count in char_counts.items()}

    # Median value per feature (legitimate rows) used as a fallback when a
    # webpage-derived feature cannot be computed for a single-URL lookup.
    feature_defaults = df.loc[df["label"] == 1, feature_columns].median().to_dict()

    # Correlation matrix over the top-importance features, for the EDA/heatmap plot.
    top_features = list(metrics["feature_importances"].keys())[:15]
    correlation = df[top_features].corr().round(3)
    correlation_artifact = {"features": top_features, "matrix": correlation.values.tolist()}

    joblib.dump(model, MODELS_DIR / "model.joblib")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(feature_columns, indent=2))
    (MODELS_DIR / "tld_lookup.json").write_text(json.dumps(tld_lookup, indent=2))
    (MODELS_DIR / "legit_domains.json").write_text(json.dumps(legit_domains))
    (MODELS_DIR / "char_freq.json").write_text(json.dumps(char_freq, indent=2))
    (MODELS_DIR / "feature_defaults.json").write_text(json.dumps(feature_defaults, indent=2))
    (MODELS_DIR / "correlation.json").write_text(json.dumps(correlation_artifact, indent=2))
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Trained on {len(X_train)} rows, tested on {len(X_test)} rows")
    print(f"Accuracy: {metrics['accuracy']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Artifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    train_and_save()
