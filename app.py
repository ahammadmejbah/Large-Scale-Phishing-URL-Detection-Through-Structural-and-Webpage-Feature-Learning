"""Streamlit app for the PhiUSIIL Phishing URL Detection dataset.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.feature_extraction import fetch_webpage_features, lexical_features
from src.train_nlp_model import tokenize_title  # noqa: F401 - required for unpickling the TF-IDF vectorizer

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "Datasets" / "PhiUSIIL_Phishing_URL_Dataset.csv"

ICON_SHIELD = ":material/shield_person:"
ICON_HOME = ":material/home:"
ICON_SEARCH = ":material/search:"
ICON_TABLE = ":material/table_view:"
ICON_CHART = ":material/bar_chart:"
ICON_INSIGHTS = ":material/insights:"
ICON_NLP = ":material/text_fields:"
ICON_CHECK = ":material/check_circle:"
ICON_WARN = ":material/warning:"
ICON_ERROR = ":material/gpp_maybe:"
ICON_INFO = ":material/info:"

st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon=ICON_SHIELD,
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "model.joblib")
    feature_columns = json.loads((MODELS_DIR / "feature_columns.json").read_text())
    tld_lookup = json.loads((MODELS_DIR / "tld_lookup.json").read_text())
    legit_domains = json.loads((MODELS_DIR / "legit_domains.json").read_text())
    char_freq = json.loads((MODELS_DIR / "char_freq.json").read_text())
    feature_defaults = json.loads((MODELS_DIR / "feature_defaults.json").read_text())
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    correlation = json.loads((MODELS_DIR / "correlation.json").read_text())
    return (
        model, feature_columns, tld_lookup, legit_domains, char_freq,
        feature_defaults, metrics, correlation,
    )


@st.cache_resource
def load_nlp_artifacts():
    vectorizer = joblib.load(MODELS_DIR / "nlp_vectorizer.joblib")
    nlp_model = joblib.load(MODELS_DIR / "nlp_model.joblib")
    nlp_metrics = json.loads((MODELS_DIR / "nlp_metrics.json").read_text())
    word_frequencies = json.loads((MODELS_DIR / "word_frequencies.json").read_text())
    title_lengths = json.loads((MODELS_DIR / "title_lengths.json").read_text())
    return vectorizer, nlp_model, nlp_metrics, word_frequencies, title_lengths


@st.cache_data
def load_dataset_sample(n: int = 20000) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if len(df) > n:
        df = df.sample(n, random_state=42)
    return df


def artifacts_available() -> bool:
    required = [
        "model.joblib", "feature_columns.json", "tld_lookup.json",
        "legit_domains.json", "char_freq.json", "feature_defaults.json", "metrics.json",
        "correlation.json",
    ]
    return all((MODELS_DIR / f).exists() for f in required)


def nlp_artifacts_available() -> bool:
    required = [
        "nlp_vectorizer.joblib", "nlp_model.joblib", "nlp_metrics.json",
        "word_frequencies.json", "title_lengths.json",
    ]
    return all((MODELS_DIR / f).exists() for f in required)


def build_full_feature_row(url: str, use_live_fetch: bool, artifacts) -> tuple[dict, dict]:
    _, feature_columns, tld_lookup, legit_domains, char_freq, feature_defaults, _, _ = artifacts

    lex_features, host, tld = lexical_features(url, tld_lookup, char_freq, legit_domains)

    row = dict(feature_defaults)
    row.update(lex_features)

    info = {"host": host, "tld": tld, "live_fetch_used": False, "live_fetch_error": None}

    if use_live_fetch:
        try:
            web_features, title = fetch_webpage_features(url)
            row.update(web_features)
            info["live_fetch_used"] = True
            info["title"] = title
        except Exception as exc:  # noqa: BLE001 - surface any fetch failure (network, SSRF guard, parsing)
            info["live_fetch_error"] = str(exc)

    ordered_row = {col: row.get(col, 0) for col in feature_columns}
    return ordered_row, info


def gauge_chart(value: float, title: str) -> go.Figure:
    color = "#2e7d32" if value >= 50 else "#c62828"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%"},
            title={"text": title},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#ffcdd2"},
                    {"range": [50, 100], "color": "#c8e6c9"},
                ],
            },
        )
    )
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
    return fig


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"## {ICON_SHIELD} Phishing URL Detection")
st.sidebar.caption("PhiUSIIL dataset · machine learning, data science & NLP")
page = st.sidebar.radio(
    "Navigate",
    [
        f"{ICON_HOME} Overview",
        f"{ICON_SEARCH} Check a URL",
        f"{ICON_TABLE} Batch CSV Prediction",
        f"{ICON_CHART} Dataset Explorer",
        f"{ICON_INSIGHTS} Model Performance",
        f"{ICON_NLP} NLP Insights",
    ],
)

if not artifacts_available():
    st.error(
        "Model artifacts not found in `models/`. Train the model first by running:\n\n"
        "`python -m src.train_model`",
        icon=ICON_ERROR,
    )
    st.stop()

artifacts = load_artifacts()
(
    model, feature_columns, tld_lookup, legit_domains, char_freq,
    feature_defaults, metrics, correlation,
) = artifacts

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page.endswith("Overview"):
    st.title(f"{ICON_SHIELD} Phishing URL Detection — PhiUSIIL Dataset")
    st.markdown(
        """
The **PhiUSIIL Phishing URL Dataset** contains **235,795** URLs
(**134,850** legitimate, **100,945** phishing) described by 50+ structural
(URL-based) and webpage-based features, plus the raw page `<title>` text.

This app combines three complementary data-science approaches:
- **Structural / tabular ML** — a Random Forest over 50 numeric URL & webpage features.
- **NLP** — a TF-IDF + Logistic Regression classifier trained on webpage title text.
- **Exploratory data analysis** — distribution, correlation, and importance plots.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{ICON_TABLE} Total URLs", "235,795")
    col2.metric(f"{ICON_CHECK} Legitimate", "134,850")
    col3.metric(f"{ICON_WARN} Phishing", "100,945")
    col4.metric(f"{ICON_INSIGHTS} Structural model ROC-AUC", f"{metrics['roc_auc']:.4f}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        counts = pd.Series({"legitimate": 134850, "phishing": 100945})
        fig = px.pie(values=counts.values, names=counts.index, hole=0.45,
                     title="Class balance", color=counts.index,
                     color_discrete_map={"legitimate": "#2e7d32", "phishing": "#c62828"})
        st.plotly_chart(fig, width='stretch')
    with c2:
        feature_groups = pd.Series({
            "URL/structural": 21,
            "Webpage/HTML": 26,
            "Text (title)": 1,
            "Identifiers (excluded)": 5,
        })
        fig2 = px.bar(x=feature_groups.index, y=feature_groups.values,
                      labels={"x": "Feature group", "y": "Column count"},
                      title="Dataset feature groups", color=feature_groups.index)
        st.plotly_chart(fig2, width='stretch')

    st.info(
        "Model: Random Forest trained on all structural & webpage features "
        "(excluding raw identifiers like FILENAME/URL/Domain/TLD/Title).",
        icon=ICON_INFO,
    )

# ---------------------------------------------------------------------------
# Check a URL
# ---------------------------------------------------------------------------
elif page.endswith("Check a URL"):
    st.title(f"{ICON_SEARCH} Check a URL")
    st.caption(
        "Lexical/structural features are computed directly from the URL. "
        "Webpage features (title, forms, scripts, images...) require fetching the live page."
    )

    url_input = st.text_input("Enter a URL", placeholder="https://example.com")
    use_live_fetch = st.checkbox(
        "Fetch the live page to extract webpage features (recommended for accuracy)",
        value=False,
        help=(
            "Performs an HTTP GET to the URL with SSRF protections (blocks private/internal "
            "addresses, limits redirects and response size). If disabled or the fetch fails, "
            "webpage features fall back to dataset-derived median values."
        ),
    )

    if st.button("Analyze", type="primary", icon=ICON_SEARCH, disabled=not url_input.strip()):
        with st.spinner("Analyzing URL..."):
            row, info = build_full_feature_row(url_input, use_live_fetch, artifacts)
            X = pd.DataFrame([row], columns=feature_columns)
            proba = model.predict_proba(X)[0]
            pred = model.predict(X)[0]

        phishing_prob = proba[0]
        legit_prob = proba[1]

        if pred == 1:
            st.success(f"Likely **Legitimate** ({legit_prob:.1%} confidence)", icon=ICON_CHECK)
        else:
            st.error(f"Likely **Phishing** ({phishing_prob:.1%} confidence)", icon=ICON_ERROR)

        c1, c2 = st.columns(2)
        c1.plotly_chart(gauge_chart(legit_prob * 100, "Legitimate probability"), width='stretch')
        c2.plotly_chart(gauge_chart(phishing_prob * 100, "Phishing probability"), width='stretch')

        if info["live_fetch_error"]:
            st.warning(
                f"Could not fetch the live page ({info['live_fetch_error']}). "
                "Webpage-based features were filled with dataset median values.",
                icon=ICON_WARN,
            )
        elif not use_live_fetch:
            st.warning(
                "Live page not fetched — webpage-based features (title, forms, scripts, etc.) "
                "use dataset median fallback values, which may reduce accuracy.",
                icon=ICON_WARN,
            )
        elif info["live_fetch_used"]:
            st.caption(f"Fetched page title: \"{info.get('title', '')}\"")

        with st.expander("View computed feature vector"):
            feat_df = X.T.rename(columns={0: "value"})
            st.dataframe(feat_df, width='stretch')
            top_feats = feat_df.loc[list(metrics["feature_importances"].keys())[:10]]
            fig = px.bar(top_feats, x="value", y=top_feats.index, orientation="h",
                         title="Top-10 most important features for this URL")
            st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Batch CSV Prediction
# ---------------------------------------------------------------------------
elif page.endswith("Batch CSV Prediction"):
    st.title(f"{ICON_TABLE} Batch CSV Prediction")
    st.markdown(
        "Upload a CSV containing (at least) the following feature columns "
        "matching the PhiUSIIL schema:"
    )
    st.code(", ".join(feature_columns), language=None)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}", icon=ICON_ERROR)
            st.stop()

        missing = [c for c in feature_columns if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}", icon=ICON_ERROR)
        else:
            X = batch_df[feature_columns]
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]

            result_df = batch_df.copy()
            result_df["prediction"] = ["legitimate" if p == 1 else "phishing" for p in preds]
            result_df["legitimate_probability"] = probs

            st.success(f"Scored {len(result_df)} rows.", icon=ICON_CHECK)
            st.dataframe(result_df.head(200), width='stretch')

            c1, c2 = st.columns(2)
            with c1:
                counts = result_df["prediction"].value_counts()
                fig = px.pie(values=counts.values, names=counts.index, title="Prediction breakdown",
                             color=counts.index,
                             color_discrete_map={"legitimate": "#2e7d32", "phishing": "#c62828"})
                st.plotly_chart(fig, width='stretch')
            with c2:
                fig2 = px.histogram(result_df, x="legitimate_probability", color="prediction",
                                     nbins=40, title="Predicted probability distribution",
                                     color_discrete_map={"legitimate": "#2e7d32", "phishing": "#c62828"})
                st.plotly_chart(fig2, width='stretch')

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions as CSV", csv_bytes, file_name="predictions.csv",
                mime="text/csv", icon=":material/download:",
            )

# ---------------------------------------------------------------------------
# Dataset Explorer
# ---------------------------------------------------------------------------
elif page.endswith("Dataset Explorer"):
    st.title(f"{ICON_CHART} Dataset Explorer")
    df = load_dataset_sample()
    df["label_name"] = df["label"].map({0: "phishing", 1: "legitimate"})
    color_map = {"legitimate": "#2e7d32", "phishing": "#c62828"}

    st.caption(f"Showing a sample of {len(df):,} rows for interactive exploration.")

    class_counts = df["label_name"].value_counts()
    fig1 = px.bar(x=class_counts.index, y=class_counts.values, labels={"x": "Class", "y": "Count"},
                  title="Class distribution (sample)", color=class_counts.index,
                  color_discrete_map=color_map)
    st.plotly_chart(fig1, width='stretch')

    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]

    c1, c2 = st.columns(2)
    with c1:
        selected = st.selectbox("Feature to explore (histogram)", numeric_cols,
                                 index=numeric_cols.index("URLSimilarityIndex"))
        fig2 = px.histogram(
            df, x=selected, color="label_name", barmode="overlay", nbins=50,
            title=f"Distribution of {selected} by class", color_discrete_map=color_map,
        )
        st.plotly_chart(fig2, width='stretch')
    with c2:
        fig3 = px.box(df, x="label_name", y=selected, color="label_name",
                      title=f"{selected} spread by class", color_discrete_map=color_map)
        st.plotly_chart(fig3, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        x_feat = st.selectbox("Scatter X", numeric_cols, index=numeric_cols.index("URLSimilarityIndex"))
    with c4:
        y_feat = st.selectbox("Scatter Y", numeric_cols, index=numeric_cols.index("NoOfExternalRef"))
    fig4 = px.scatter(df.sample(min(3000, len(df)), random_state=1), x=x_feat, y=y_feat,
                       color="label_name", opacity=0.6, title=f"{x_feat} vs {y_feat}",
                       color_discrete_map=color_map)
    st.plotly_chart(fig4, width='stretch')

    st.subheader("Correlation heatmap (top 15 most important features)")
    corr_df = pd.DataFrame(correlation["matrix"], index=correlation["features"], columns=correlation["features"])
    fig5 = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                      title="Feature correlation matrix")
    st.plotly_chart(fig5, width='stretch')

    st.subheader("Raw sample")
    st.dataframe(df.drop(columns=["label_name"]).head(100), width='stretch')

# ---------------------------------------------------------------------------
# Model Performance
# ---------------------------------------------------------------------------
elif page.endswith("Model Performance"):
    st.title(f"{ICON_INSIGHTS} Model Performance")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['precision']:.4f}")
    c3.metric("Recall", f"{metrics['recall']:.4f}")
    c4.metric("F1", f"{metrics['f1']:.4f}")
    c5.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")

    st.caption(f"Evaluated on a held-out test set of {metrics['n_test']:,} rows "
               f"(trained on {metrics['n_train']:,} rows).")

    c1, c2 = st.columns(2)
    with c1:
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual: phishing", "Actual: legitimate"],
                              columns=["Predicted: phishing", "Predicted: legitimate"])
        fig_cm = px.imshow(cm_df, text_auto=True, title="Confusion Matrix", color_continuous_scale="Blues")
        st.plotly_chart(fig_cm, width='stretch')
    with c2:
        fi = pd.Series(metrics["feature_importances"]).sort_values(ascending=True).tail(15)
        fig_fi = px.bar(x=fi.values, y=fi.index, orientation="h", title="Top 15 feature importances")
        st.plotly_chart(fig_fi, width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        roc = metrics["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC curve"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                      line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title=f"ROC curve (AUC = {metrics['roc_auc']:.4f})",
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, width='stretch')
    with c4:
        pr = metrics["pr_curve"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR curve"))
        fig_pr.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, width='stretch')

# ---------------------------------------------------------------------------
# NLP Insights
# ---------------------------------------------------------------------------
elif page.endswith("NLP Insights"):
    st.title(f"{ICON_NLP} NLP Insights — Webpage Title Classifier")
    st.caption(
        "A separate TF-IDF + Logistic Regression model trained only on webpage "
        "`<title>` text, to see how much signal text alone carries."
    )

    if not nlp_artifacts_available():
        st.error(
            "NLP artifacts not found in `models/`. Train the NLP model first by running:\n\n"
            "`python -m src.train_nlp_model`",
            icon=ICON_ERROR,
        )
        st.stop()

    vectorizer, nlp_model, nlp_metrics, word_frequencies, title_lengths = load_nlp_artifacts()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{nlp_metrics['accuracy']:.4f}")
    c2.metric("Precision", f"{nlp_metrics['precision']:.4f}")
    c3.metric("Recall", f"{nlp_metrics['recall']:.4f}")
    c4.metric("F1", f"{nlp_metrics['f1']:.4f}")
    c5.metric("ROC-AUC", f"{nlp_metrics['roc_auc']:.4f}")

    st.subheader("Try it")
    title_input = st.text_input("Enter a webpage title", placeholder="Login to your account - Secure Banking")
    if st.button("Classify title", icon=ICON_NLP, disabled=not title_input.strip()):
        vec = vectorizer.transform([title_input])
        pred = nlp_model.predict(vec)[0]
        proba = nlp_model.predict_proba(vec)[0]
        label = "Legitimate" if pred == 1 else "Phishing"
        icon = ICON_CHECK if pred == 1 else ICON_ERROR
        st.success(f"Predicted: **{label}** ({proba[pred]:.1%} confidence)", icon=icon)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        cm = nlp_metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual: phishing", "Actual: legitimate"],
                              columns=["Predicted: phishing", "Predicted: legitimate"])
        fig_cm = px.imshow(cm_df, text_auto=True, title="NLP Confusion Matrix", color_continuous_scale="Purples")
        st.plotly_chart(fig_cm, width='stretch')
    with c2:
        lengths_df = pd.concat([
            pd.DataFrame({"length": title_lengths["legitimate"], "class": "legitimate"}),
            pd.DataFrame({"length": title_lengths["phishing"], "class": "phishing"}),
        ])
        fig_len = px.histogram(lengths_df, x="length", color="class", barmode="overlay", nbins=60,
                                range_x=[0, 100], title="Title length distribution by class",
                                color_discrete_map={"legitimate": "#2e7d32", "phishing": "#c62828"})
        st.plotly_chart(fig_len, width='stretch')

    st.subheader("Most predictive terms (TF-IDF + Logistic Regression coefficients)")
    c3, c4 = st.columns(2)
    with c3:
        legit_terms = pd.DataFrame(nlp_metrics["top_terms_legitimate"])
        fig_legit = px.bar(legit_terms.sort_values("weight"), x="weight", y="term", orientation="h",
                            title="Top terms → legitimate", color_discrete_sequence=["#2e7d32"])
        st.plotly_chart(fig_legit, width='stretch')
    with c4:
        phishing_terms = pd.DataFrame(nlp_metrics["top_terms_phishing"])
        phishing_terms["abs_weight"] = phishing_terms["weight"].abs()
        fig_phish = px.bar(phishing_terms.sort_values("abs_weight"), x="weight", y="term", orientation="h",
                            title="Top terms → phishing", color_discrete_sequence=["#c62828"])
        st.plotly_chart(fig_phish, width='stretch')

    st.subheader("Most frequent words in titles")
    c5, c6 = st.columns(2)
    with c5:
        legit_words = pd.DataFrame(word_frequencies["legitimate"], columns=["word", "count"])
        fig_w1 = px.bar(legit_words.sort_values("count"), x="count", y="word", orientation="h",
                         title="Legitimate titles", color_discrete_sequence=["#2e7d32"])
        st.plotly_chart(fig_w1, width='stretch')
    with c6:
        phishing_words = pd.DataFrame(word_frequencies["phishing"], columns=["word", "count"])
        fig_w2 = px.bar(phishing_words.sort_values("count"), x="count", y="word", orientation="h",
                         title="Phishing titles", color_discrete_sequence=["#c62828"])
        st.plotly_chart(fig_w2, width='stretch')

