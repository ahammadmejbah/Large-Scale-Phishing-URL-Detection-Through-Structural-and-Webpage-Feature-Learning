"""Train an NLP model (TF-IDF + Logistic Regression) on webpage <title> text
to classify legitimate vs phishing pages from text alone.

Produces artifacts under `models/`:
  - nlp_vectorizer.joblib   fitted TfidfVectorizer
  - nlp_model.joblib        trained LogisticRegression classifier
  - nlp_metrics.json        evaluation metrics + top predictive terms
  - word_frequencies.json   most common words per class (for bar-chart plots)
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "Datasets" / "PhiUSIIL_Phishing_URL_Dataset.csv"
MODELS_DIR = ROOT_DIR / "models"

TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")


def tokenize_title(text: str) -> list[str]:
    """Tokenizer used by the TF-IDF vectorizer (must stay importable for unpickling)."""
    return [t.lower() for t in TOKEN_RE.findall(str(text))]


def _top_words(titles: pd.Series, top_n: int = 25) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for title in titles:
        counter.update(w for w in tokenize_title(title) if w not in ENGLISH_STOP_WORDS)
    return counter.most_common(top_n)


def train_and_save() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    titles = df["Title"].astype(str)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        titles, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_title,
        token_pattern=None,
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    y_proba = clf.predict_proba(X_test_vec)[:, 1]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    top_legit_idx = np.argsort(coefs)[-25:][::-1]
    top_phishing_idx = np.argsort(coefs)[:25]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "top_terms_legitimate": [
            {"term": feature_names[i], "weight": float(coefs[i])} for i in top_legit_idx
        ],
        "top_terms_phishing": [
            {"term": feature_names[i], "weight": float(coefs[i])} for i in top_phishing_idx
        ],
    }

    word_frequencies = {
        "legitimate": _top_words(df.loc[df["label"] == 1, "Title"]),
        "phishing": _top_words(df.loc[df["label"] == 0, "Title"]),
    }

    title_lengths = {
        "legitimate": df.loc[df["label"] == 1, "Title"].astype(str).str.len().tolist()[:20000],
        "phishing": df.loc[df["label"] == 0, "Title"].astype(str).str.len().tolist()[:20000],
    }

    joblib.dump(vectorizer, MODELS_DIR / "nlp_vectorizer.joblib")
    joblib.dump(clf, MODELS_DIR / "nlp_model.joblib")
    (MODELS_DIR / "nlp_metrics.json").write_text(json.dumps(metrics, indent=2))
    (MODELS_DIR / "word_frequencies.json").write_text(json.dumps(word_frequencies, indent=2))
    (MODELS_DIR / "title_lengths.json").write_text(json.dumps(title_lengths))

    print(f"NLP model trained on {len(X_train)} titles, tested on {len(X_test)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}  ROC-AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    train_and_save()
