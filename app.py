"""Streamlit app for the PhiUSIIL Phishing URL Detection dataset.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
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

def make_analysis_chart(kind: str, values: list[float], labels: list[str] | None = None):
    """Create a compact plotly figure for the overview analysis cards."""
    if labels is None:
        labels = [str(i + 1) for i in range(len(values))]

    if kind == "bar":
        fig = px.bar(
            x=labels,
            y=values,
            color=values,
            color_continuous_scale="RdYlGn_r",
            range_color=[0, max(values) * 1.2],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
    elif kind == "line":
        fig = px.line(x=labels, y=values, markers=True)
        fig.update_traces(line=dict(color="#111111", width=2.5), marker=dict(size=6, color="#111111"))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
    else:
        fig = px.pie(names=labels, values=values, hole=0.5)
        fig.update_traces(marker=dict(colors=["#111111", "#d4d4d4", "#9ca3af", "#ef4444"]))
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
    return fig


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
if page.endswith("Overview"):
    st.markdown(
        """
        <style>
            .hero-shell {
                position: relative;
                overflow: hidden;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 26px;
                padding: 2.8rem 2rem 2rem 2rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
            }
            .hero-shell::before,
            .hero-shell::after {
                content: "";
                position: absolute;
                border-radius: 50%;
                filter: blur(18px);
                opacity: 0.2;
            }
            .hero-shell::before {
                width: 220px;
                height: 220px;
                right: -20px;
                top: -30px;
                background: rgba(59, 130, 246, 0.28);
            }
            .hero-shell::after {
                width: 220px;
                height: 220px;
                left: -20px;
                bottom: -30px;
                background: rgba(34, 197, 94, 0.20);
            }
            .hero-inner {
                position: relative;
                z-index: 1;
                max-width: 1100px;
                margin: 0 auto;
                text-align: center;
            }
            .hero-kicker {
                display: inline-block;
                font-size: 0.75rem;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                color: #475569;
                font-weight: 700;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 999px;
                padding: 0.45rem 0.9rem;
                margin-bottom: 1rem;
            }
            .hero-title {
                font-size: clamp(2.3rem, 4vw, 4.5rem);
                line-height: 1.04;
                font-weight: 900;
                margin: 0 0 1rem 0;
                color: #0f172a;
            }
            .hero-title .accent {
                color: #7dd3fc;
            }
            .hero-subtitle {
                font-size: 1.08rem;
                line-height: 1.8;
                color: #475569;
                margin: 0 auto;
                max-width: 820px;
            }
            .hero-actions {
                display: flex;
                justify-content: center;
                gap: 0.8rem;
                flex-wrap: wrap;
                margin-top: 1.7rem;
            }
            .hero-button {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.8rem 1.35rem;
                border-radius: 14px;
                font-weight: 700;
                color: #0f172a !important;
                text-decoration: none !important;
                border: 1px solid #cbd5e1;
            }
            .hero-button:visited,
            .hero-button:active,
            .hero-button:focus {
                color: #0f172a !important;
                text-decoration: none !important;
            }
            .hero-primary {
                background: #0f172a;
                color: #ffffff !important;
            }
            .hero-primary:hover,
            .hero-primary:active,
            .hero-primary:visited,
            .hero-primary:focus {
                background: #0f172a !important;
                color: #ffffff !important;
                opacity: 1;
                transform: none;
                filter: none;
                box-shadow: none;
                transition: none;
            }
            .hero-secondary {
                background: #f8fafc;
                color: #0f172a !important;
            }
            .hero-mini-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.85rem;
                margin-top: 2rem;
            }
            .hero-mini-card {
                text-align: left;
                padding: 1rem;
                border-radius: 16px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                box-shadow: 0 12px 24px rgba(15, 23, 42, 0.06);
            }
            .hero-mini-card .label {
                display: block;
                color: #64748b;
                font-size: 0.72rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                margin-bottom: 0.45rem;
            }
            .hero-mini-card .value {
                color: #0f172a;
                font-size: 1.45rem;
                font-weight: 800;
            }
            .trust-strip {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 0.7rem;
                margin-top: 1.5rem;
            }
            .trust-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 999px;
                padding: 0.5rem 0.85rem;
                color: #334155;
                font-size: 0.78rem;
                font-weight: 600;
            }
            .trust-pill .dot {
                width: 0.55rem;
                height: 0.55rem;
                border-radius: 50%;
                background: linear-gradient(135deg, #34d399, #60a5fa);
                box-shadow: 0 0 10px rgba(96, 165, 250, 0.9);
            }
            .feature-badges {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 0.6rem;
                margin-top: 1.6rem;
            }
            .feature-badge {
                display: inline-block;
                padding: 0.52rem 0.9rem;
                border-radius: 999px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                color: #334155;
                font-weight: 600;
            }
            .stack-shell {
                margin-top: 1.5rem;
                margin-bottom: 1.5rem;
            }
            .stack-title {
                text-align: center;
                font-size: 1.8rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 1rem;
            }
            .stack-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
            }
            .stack-card {
                border: 1px solid rgba(15, 23, 42, 0.08);
                background: rgba(255,255,255,0.8);
                border-radius: 20px;
                padding: 1.25rem;
                box-shadow: 0 16px 32px rgba(15, 23, 42, 0.04);
            }
            .stack-card .index {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2.1rem;
                height: 2.1rem;
                border-radius: 50%;
                background: #111827;
                color: white;
                font-weight: 700;
                margin-bottom: 0.75rem;
            }
            .stack-card h4 {
                margin: 0 0 0.5rem 0;
                color: #0f172a;
                font-size: 1.2rem;
            }
            .stack-card p {
                margin: 0;
                color: #475569;
                line-height: 1.7;
            }
            .tab-shell {
                margin-top: 1.2rem;
            }
            .section-heading {
                text-align: center;
                font-size: 2.2rem;
                font-weight: 900;
                color: #0f172a;
                margin: 1.7rem 0 0.5rem 0;
            }
            .section-subtitle {
                text-align: center;
                color: #475569;
                font-size: 1rem;
                margin-bottom: 1.2rem;
            }
            .analysis-card {
                background: rgba(255,255,255,0.82);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 12px 26px rgba(15, 23, 42, 0.04);
                height: 100%;
            }
            .analysis-card .badgie {
                width: 2.2rem;
                height: 2.2rem;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                border-radius: 50%;
                background: #000000;
                color: white;
                font-weight: 700;
                margin-bottom: 0.9rem;
            }
            .analysis-card h4 {
                margin: 0 0 0.55rem 0;
                color: #0f172a;
                font-size: 1.05rem;
            }
            .analysis-card .summary {
                color: #111827;
                font-weight: 600;
                line-height: 1.6;
                margin-bottom: 0.85rem;
            }
            .analysis-card .detail {
                color: #475569;
                line-height: 1.65;
                font-size: 0.9rem;
                margin-bottom: 0.9rem;
            }
            .analysis-card .plot-box {
                border-radius: 12px;
                background: rgba(15, 23, 42, 0.02);
                padding: 0.25rem;
            }
            .stats-card {
                border: 1px solid rgba(148, 163, 184, 0.22);
                border-radius: 18px;
                padding: 1.2rem 1rem;
                background: rgba(255,255,255,0.7);
                box-shadow: 0 6px 18px rgba(15,23,42,0.04);
            }
            .mini-label {
                display: block;
                font-size: 0.72rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #64748b;
                margin-bottom: 0.5rem;
            }
            .mini-value {
                font-size: 1.8rem;
                font-weight: 800;
                color: #0f172a;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-inner">
                <div class="hero-kicker">AI Security Intelligence</div>
                <h1 class="hero-title">Detect <span class="accent">phishing URLs</span> before they reach users.</h1>
                <p class="hero-subtitle">
                    This platform combines structural signal analysis, live webpage intelligence, and NLP-driven title scoring
                    to classify malicious URLs with explainable insights, transparent probabilities, and actionable risk signals.
                </p>
                <div class="hero-actions">
                    <a class="hero-button hero-primary" href="#check-a-url">Analyze a URL</a>
                    <a class="hero-button hero-secondary" href="#dataset-explorer">Explore the dataset</a>
                </div>
                <div class="feature-badges">
                    <span class="feature-badge">Random Forest</span>
                    <span class="feature-badge">TF-IDF + Logistic Regression</span>
                    <span class="feature-badge">Live Webpage Features</span>
                    <span class="feature-badge">Explainable Risk Scores</span>
                </div>
                <div class="hero-mini-grid">
                    <div class="hero-mini-card">
                        <span class="label">Urls</span>
                        <span class="value">235,795</span>
                    </div>
                    <div class="hero-mini-card">
                        <span class="label">Legit</span>
                        <span class="value">134,850</span>
                    </div>
                    <div class="hero-mini-card">
                        <span class="label">Phishing</span>
                        <span class="value">100,945</span>
                    </div>
                    <div class="hero-mini-card">
                        <span class="label">AUC</span>
                        <span class="value">{metrics['roc_auc']:.4f}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="trust-strip">
            <div class="trust-pill"><span class="dot"></span> Live URL scoring</div>
            <div class="trust-pill"><span class="dot"></span> NLP title analysis</div>
            <div class="trust-pill"><span class="dot"></span> Structural risk profiling</div>
            <div class="trust-pill"><span class="dot"></span> Real-time detection dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="stack-shell">
            <div class="stack-title">Why phishing detection matters</div>
            <div class="stack-grid">
                <div class="stack-card">
                    <div class="index">1</div>
                    <h4>Threat prevention</h4>
                    <p>Phishing attacks rely on urgency, impersonation, and trust manipulation. Early detection prevents credential theft and fraud.</p>
                </div>
                <div class="stack-card">
                    <div class="index">2</div>
                    <h4>Brand protection</h4>
                    <p>Fake login pages often mimic trusted brands. Models trained on real-world phishing patterns detect visual and lexical spoofing.</p>
                </div>
                <div class="stack-card">
                    <div class="index">3</div>
                    <h4>Operational intelligence</h4>
                    <p>By combining structure, webpage behavior, and semantic signals, security teams gain explainable and actionable alerts.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="stack-shell"><div class="stack-title">How the detection stack works</div></div>', unsafe_allow_html=True)
    sp1, sp2, sp3 = st.columns(3)
    with sp1:
        st.markdown(
            """
            <div class="stack-card">
                <div class="index">1</div>
                <h4>URL signal extraction</h4>
                <p>Lexical and structural features detect suspicious length, entropy, token tricks, and deceptive domain patterns.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sp2:
        st.markdown(
            """
            <div class="stack-card">
                <div class="index">2</div>
                <h4>Webpage intelligence</h4>
                <p>Live fetching adds title, form, image, script, and redirect signals that reveal credential-harvesting behavior.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with sp3:
        st.markdown(
            """
            <div class="stack-card">
                <div class="index">3</div>
                <h4>Risk scoring</h4>
                <p>Combining tabular ML and NLP creates a final probability estimate, making the model explainable and practical.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    overview_df = load_dataset_sample(n=20000)
    overview_df = overview_df.assign(
        Class=overview_df["label"].map({0: "Phishing", 1: "Legitimate"})
    )
    class_order = ["Legitimate", "Phishing"]
    sample_class_counts = (
        overview_df["Class"].value_counts().reindex(class_order, fill_value=0)
    )

    dataset_section, feature_section, model_section = (
        st.container(), st.container(), st.container()
    )

    with dataset_section:
        st.header("Dataset analytics")
        st.caption(
            f"Dataset totals are shown above. Distribution and feature statistics below "
            f"use a reproducible {len(overview_df):,}-row sample (random seed 42)."
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="stats-card"><span class="mini-label">Total URLs</span><div class="mini-value">235,795</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="stats-card"><span class="mini-label">Legitimate</span><div class="mini-value">134,850</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="stats-card"><span class="mini-label">Phishing</span><div class="mini-value">100,945</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stats-card"><span class="mini-label">ROC-AUC</span><div class="mini-value">{metrics["roc_auc"]:.4f}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-heading">Class and URL profile</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-subtitle">Observed patterns in the reproducible dataset sample.</div>', unsafe_allow_html=True)
        chart1, chart2 = st.columns(2)
        with chart1:
            fig = px.pie(
                values=sample_class_counts.values,
                names=sample_class_counts.index,
                hole=0.45,
                title="Sample class distribution",
                color=sample_class_counts.index,
                color_discrete_map={"Legitimate": "#238636", "Phishing": "#d73a49"},
            )
            st.plotly_chart(fig, width='stretch', key='overview_class_balance')
        with chart2:
            feature_groups = pd.Series({
                "URL / structural": 22,
                "Webpage / HTML": max(len(feature_columns) - 22, 0),
                "Excluded identifiers": 6,
            })
            fig2 = px.bar(
                x=feature_groups.index,
                y=feature_groups.values,
                labels={"x": "Feature group", "y": "Column count"},
                title="Model input and excluded columns",
                color=feature_groups.index,
            )
            st.plotly_chart(fig2, width='stretch', key='overview_feature_groups')

        plot1, plot2 = st.columns(2)
        with plot1:
            fig = px.histogram(
                overview_df,
                x="URLLength",
                color="Class",
                color_discrete_map={"Legitimate": "#238636", "Phishing": "#d73a49"},
                barmode="overlay",
                opacity=0.65,
                nbins=50,
                marginal="box",
                title="URL length distribution by class",
            )
            fig.update_layout(xaxis_title="URL length (characters)", yaxis_title="URL count")
            st.plotly_chart(fig, width="stretch", key="overview_url_length_distribution")
        with plot2:
            fig = px.scatter(
                overview_df,
                x="DomainLength",
                y="URLLength",
                color="Class",
                opacity=0.45,
                color_discrete_map={"Legitimate": "#238636", "Phishing": "#d73a49"},
                title="URL length vs. domain length",
                labels={"DomainLength": "Domain length", "URLLength": "URL length"},
            )
            st.plotly_chart(fig, width="stretch", key="overview_url_domain_scatter")

        st.subheader("Descriptive statistics by class")
        summary_features = [
            "URLLength", "DomainLength", "NoOfSubDomain",
            "NoOfObfuscatedChar", "NoOfExternalRef", "NoOfJS",
        ]
        summary_features = [name for name in summary_features if name in overview_df]
        summary = overview_df.groupby("Class")[summary_features].agg(["mean", "median", "std"])
        summary.columns = [f"{feature} {stat}" for feature, stat in summary.columns]
        st.dataframe(summary.reindex(class_order).style.format("{:,.2f}"), width="stretch")

        st.subheader("Class composition")
        composition = pd.DataFrame({
            "Class": class_order,
            "Sample URLs": [int(sample_class_counts[label]) for label in class_order],
            "Sample share (%)": [
                sample_class_counts[label] / len(overview_df) * 100 for label in class_order
            ],
        })
        st.dataframe(composition.style.format({"Sample share (%)": "{:.2f}%"}), width="stretch", hide_index=True)

        with st.expander("Column-level data quality and coverage"):
            column_profile = pd.DataFrame({
                "Column": overview_df.columns,
                "Data type": overview_df.dtypes.astype(str).values,
                "Missing rows": overview_df.isna().sum().values,
                "Missing (%)": overview_df.isna().mean().mul(100).values,
                "Unique values in sample": overview_df.nunique(dropna=True).values,
            }).sort_values(["Missing rows", "Column"], ascending=[False, True])
            st.dataframe(
                column_profile.style.format({"Missing (%)": "{:.2f}%"}),
                width="stretch",
                hide_index=True,
            )
            numeric_profile = overview_df.select_dtypes(include="number").drop(
                columns=["label"], errors="ignore"
            ).describe(percentiles=[0.25, 0.5, 0.75]).T
            numeric_profile.index.name = "Feature"
            numeric_profile = numeric_profile.rename(columns={
                "count": "Non-missing rows", "25%": "Q1", "50%": "Median", "75%": "Q3"
            })
            st.markdown("**Numeric feature distribution**")
            st.dataframe(numeric_profile.style.format("{:,.3f}"), width="stretch")

        binary_features = [
            "IsHTTPS", "HasObfuscation", "IsDomainIP", "HasPasswordField",
            "HasExternalFormSubmit", "HasHiddenFields",
        ]
        binary_features = [name for name in binary_features if name in overview_df]
        if binary_features:
            prevalence = overview_df.groupby("Class")[binary_features].mean().mul(100)
            prevalence_long = prevalence.rename_axis("Class").reset_index().melt(
                id_vars="Class", var_name="Signal", value_name="Sample prevalence (%)"
            )
            fig = px.bar(
                prevalence_long,
                x="Signal",
                y="Sample prevalence (%)",
                color="Class",
                barmode="group",
                color_discrete_map={"Legitimate": "#238636", "Phishing": "#d73a49"},
                title="Prevalence of selected binary signals",
            )
            st.plotly_chart(fig, width="stretch", key="overview_signal_prevalence")

        if "TLD" in overview_df:
            st.subheader("Most common TLDs in the sample")
            tld_sample = overview_df.assign(TLD=overview_df["TLD"].fillna("(missing)").astype(str))
            tld_summary = tld_sample.groupby("TLD")["label"].agg(
                URLs="size",
                Phishing=lambda labels: int((labels == 0).sum()),
                Legitimate=lambda labels: int((labels == 1).sum()),
            )
            tld_summary["Phishing rate (%)"] = (
                tld_summary["Phishing"] / tld_summary["URLs"] * 100
            )
            st.dataframe(
                tld_summary.nlargest(15, "URLs").style.format({"Phishing rate (%)": "{:.2f}%"}),
                width="stretch",
            )

    with feature_section:
        st.header("Feature analysis")
        st.subheader("Measured feature differences")
        st.caption("Bars show sample medians for each class; binary features are shown as prevalence percentages.")
        analysis_items = []
        feature_specs = [
            ("URL length", "URLLength", "median", "characters"),
            ("Domain length", "DomainLength", "median", "characters"),
            ("Subdomain count", "NoOfSubDomain", "median", "subdomains"),
            ("Obfuscation prevalence", "HasObfuscation", "mean", "percent"),
            ("Password-field prevalence", "HasPasswordField", "mean", "percent"),
            ("External-form prevalence", "HasExternalFormSubmit", "mean", "percent"),
        ]
        for title, feature, statistic, unit in feature_specs:
            if feature not in overview_df:
                continue
            grouped = overview_df.groupby("Class")[feature]
            scale = 100 if unit == "percent" else 1
            values = [
                round(float(grouped.agg(statistic).get(label, 0)) * scale, 2)
                for label in class_order
            ]
            analysis_items.append({
                "title": title,
                "summary": f"Legitimate: {values[0]:,.2f} {unit}; phishing: {values[1]:,.2f} {unit}.",
                "detail": f"Computed from the {len(overview_df):,}-row sample. This is a descriptive comparison, not a causal finding.",
                "kind": "bar",
                "values": values,
                "labels": class_order,
            })

        comparison_rows = []
        for title, feature, _, unit in feature_specs:
            if feature not in overview_df:
                continue
            legitimate = overview_df.loc[overview_df["Class"] == "Legitimate", feature]
            phishing = overview_df.loc[overview_df["Class"] == "Phishing", feature]
            pooled_std = np.sqrt((legitimate.std() ** 2 + phishing.std() ** 2) / 2)
            scale = 100 if unit == "percent" else 1
            comparison_rows.append({
                "Feature": title,
                "Legitimate median": legitimate.median() * scale,
                "Phishing median": phishing.median() * scale,
                "Median difference (P-L)": (phishing.median() - legitimate.median()) * scale,
                "Legitimate mean": legitimate.mean() * scale,
                "Phishing mean": phishing.mean() * scale,
                "Standardized mean difference": (
                    (phishing.mean() - legitimate.mean()) / pooled_std if pooled_std else 0
                ),
            })

        cols = st.columns(2)
        for i, item in enumerate(analysis_items):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="analysis-card">
                        <div class="badgie">{i + 1}</div>
                        <h4>{item['title']}</h4>
                        <div class="summary">{item['summary']}</div>
                        <div class="detail">{item['detail']}</div>
                        <div class="plot-box">
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(make_analysis_chart(item['kind'], item['values'], item['labels']), width='stretch', config={"displayModeBar": False}, key=f"analysis_card_{i}_{item['title'].replace(' ', '_').lower()}")
                st.markdown("</div></div>", unsafe_allow_html=True)

        st.subheader("Feature comparison table")
        st.caption("Standardized mean difference is descriptive; it does not imply causation or statistical significance.")
        comparison_table = pd.DataFrame(comparison_rows).set_index("Feature")
        st.dataframe(comparison_table.style.format("{:,.3f}"), width="stretch")

        st.subheader("Binary signal prevalence gaps")
        binary_features = [
            name for name in (
                "IsHTTPS", "HasObfuscation", "IsDomainIP", "HasPasswordField",
                "HasExternalFormSubmit", "HasHiddenFields",
            ) if name in overview_df
        ]
        binary_rows = []
        for feature in binary_features:
            legitimate_rate = overview_df.loc[overview_df["Class"] == "Legitimate", feature].mean() * 100
            phishing_rate = overview_df.loc[overview_df["Class"] == "Phishing", feature].mean() * 100
            binary_rows.append({
                "Signal": feature,
                "Legitimate (%)": legitimate_rate,
                "Phishing (%)": phishing_rate,
                "Gap (percentage points)": phishing_rate - legitimate_rate,
            })
        binary_table = pd.DataFrame(binary_rows).sort_values(
            "Gap (percentage points)", key=lambda values: values.abs(), ascending=False
        )
        st.dataframe(binary_table.style.format({
            "Legitimate (%)": "{:.2f}%",
            "Phishing (%)": "{:.2f}%",
            "Gap (percentage points)": "{:+.2f}",
        }), width="stretch", hide_index=True)

        st.subheader("Model feature importance and correlation")
        importance = pd.Series(metrics["feature_importances"], name="Importance").sort_values(ascending=False)
        signal_chart, correlation_chart = st.columns(2)
        with signal_chart:
            top_importance = importance.head(15).sort_values()
            fig = px.bar(
                top_importance.rename_axis("Feature").reset_index(),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Random Forest features",
            )
            st.plotly_chart(fig, width="stretch", key="overview_feature_importance")
            importance_table = pd.DataFrame({
                "Feature": importance.index,
                "Importance": importance.values,
                "Cumulative importance": importance.cumsum().values,
            }).head(25)
            st.dataframe(importance_table.style.format({
                "Importance": "{:.4f}", "Cumulative importance": "{:.3f}"
            }), width="stretch", hide_index=True)
            structural_features = set(feature_columns[:22])
            family_rows = []
            for family, family_features in (
                ("URL / structural", structural_features),
                ("Webpage / HTML", set(feature_columns) - structural_features),
            ):
                family_importance = importance[importance.index.isin(family_features)].sum()
                family_rows.append({
                    "Feature family": family,
                    "Feature count": len(family_features),
                    "Importance sum": family_importance,
                    "Importance share (%)": family_importance * 100,
                })
            st.markdown("**Importance by feature family**")
            st.dataframe(pd.DataFrame(family_rows).style.format({
                "Importance sum": "{:.4f}", "Importance share (%)": "{:.2f}%"
            }), width="stretch", hide_index=True)
        with correlation_chart:
            corr_df = pd.DataFrame(
                correlation["matrix"],
                index=correlation["features"],
                columns=correlation["features"],
            )
            fig = px.imshow(
                corr_df,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu_r",
                title="Correlation among top model features",
            )
            st.plotly_chart(fig, width="stretch", key="overview_feature_correlation")
            correlation_pairs = [
                {
                    "Feature 1": corr_df.index[row],
                    "Feature 2": corr_df.columns[column],
                    "Correlation": corr_df.iat[row, column],
                }
                for row in range(len(corr_df.index))
                for column in range(row + 1, len(corr_df.columns))
            ]
            strongest_pairs = pd.DataFrame(correlation_pairs)
            strongest_pairs["Absolute correlation"] = strongest_pairs["Correlation"].abs()
            strongest_pairs = strongest_pairs.nlargest(20, "Absolute correlation")
            st.dataframe(strongest_pairs.style.format({
                "Correlation": "{:+.3f}", "Absolute correlation": "{:.3f}"
            }), width="stretch", hide_index=True)

    with model_section:
        st.header("Model diagnostics")
        st.subheader("Held-out model performance")
        model_results = [{"Model": "Random Forest", **{
            name: metrics[name] for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
        }}]
        if nlp_artifacts_available():
            nlp_metrics = json.loads((MODELS_DIR / "nlp_metrics.json").read_text())
            model_results.append({"Model": "Title NLP", **{
                name: nlp_metrics[name] for name in ("accuracy", "precision", "recall", "f1", "roc_auc")
            }})
        model_comparison = pd.DataFrame(model_results).set_index("Model")
        st.dataframe(model_comparison.style.format("{:.4f}"), width="stretch")
        comparison_long = model_comparison.reset_index().melt(
            id_vars="Model", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            comparison_long,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            range_y=[0, 1],
            title="Evaluation metric comparison",
        )
        st.plotly_chart(fig, width="stretch", key="overview_model_metric_comparison")

        diagnostic1, diagnostic2 = st.columns(2)
        with diagnostic1:
            matrix = metrics["confusion_matrix"]
            fig = px.imshow(
                matrix,
                text_auto=True,
                x=["Predicted phishing", "Predicted legitimate"],
                y=["Actual phishing", "Actual legitimate"],
                color_continuous_scale="Blues",
                title="Random Forest confusion matrix",
                labels={"x": "Prediction", "y": "Actual class", "color": "URLs"},
            )
            st.plotly_chart(fig, width="stretch", key="overview_confusion_matrix")
            confusion_rows = []
            for index, label in enumerate(("Phishing", "Legitimate")):
                true_positive = matrix[index][index]
                support = sum(matrix[index])
                predicted = sum(row[index] for row in matrix)
                precision = true_positive / predicted if predicted else 0
                recall = true_positive / support if support else 0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
                confusion_rows.append({
                    "Class": label,
                    "Support": support,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 score": f1,
                })
            st.dataframe(pd.DataFrame(confusion_rows).style.format({
                "Precision": "{:.4f}", "Recall": "{:.4f}", "F1 score": "{:.4f}"
            }), width="stretch", hide_index=True)
            confusion_cells = []
            class_names = ("Phishing", "Legitimate")
            for actual_index, actual_label in enumerate(class_names):
                actual_total = sum(matrix[actual_index])
                for predicted_index, predicted_label in enumerate(class_names):
                    count = matrix[actual_index][predicted_index]
                    confusion_cells.append({
                        "Actual class": actual_label,
                        "Predicted class": predicted_label,
                        "URLs": count,
                        "Share of actual class (%)": count / actual_total * 100 if actual_total else 0,
                    })
            st.dataframe(pd.DataFrame(confusion_cells).style.format({
                "Share of actual class (%)": "{:.2f}%"
            }), width="stretch", hide_index=True)
        with diagnostic2:
            roc_data = pd.DataFrame(metrics["roc_curve"])
            fig = px.line(
                roc_data,
                x="fpr",
                y="tpr",
                title=f"ROC curve (AUC {metrics['roc_auc']:.4f})",
                labels={"fpr": "False positive rate", "tpr": "True positive rate"},
            )
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line={"dash": "dash"}))
            st.plotly_chart(fig, width="stretch", key="overview_roc_curve")

        pr_data = pd.DataFrame(metrics["pr_curve"])
        fig = px.line(
            pr_data,
            x="recall",
            y="precision",
            title="Random Forest precision-recall curve",
            labels={"recall": "Recall", "precision": "Precision"},
        )
        st.plotly_chart(fig, width="stretch", key="overview_precision_recall_curve")
        st.caption(
            f"Evaluation uses the held-out test split of {metrics['n_test']:,} URLs. "
            "The dataset-sample charts above are descriptive and separate from these test metrics."
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
        c1.plotly_chart(gauge_chart(legit_prob * 100, "Legitimate probability"), width='stretch', key='url_legit_gauge')
        c2.plotly_chart(gauge_chart(phishing_prob * 100, "Phishing probability"), width='stretch', key='url_phish_gauge')

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
            st.plotly_chart(fig, width='stretch', key='top_features_for_url')

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
                st.plotly_chart(fig, width='stretch', key='batch_prediction_breakdown')
            with c2:
                fig2 = px.histogram(result_df, x="legitimate_probability", color="prediction",
                                     nbins=40, title="Predicted probability distribution",
                                     color_discrete_map={"legitimate": "#2e7d32", "phishing": "#c62828"})
                st.plotly_chart(fig2, width='stretch', key='batch_probability_distribution')

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
    st.plotly_chart(fig1, width='stretch', key='dataset_class_distribution')

    numeric_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]

    c1, c2 = st.columns(2)
    with c1:
        selected = st.selectbox("Feature to explore (histogram)", numeric_cols,
                                 index=numeric_cols.index("URLSimilarityIndex"))
        fig2 = px.histogram(
            df, x=selected, color="label_name", barmode="overlay", nbins=50,
            title=f"Distribution of {selected} by class", color_discrete_map=color_map,
        )
        st.plotly_chart(fig2, width='stretch', key=f'dataset_histogram_{selected}')
    with c2:
        fig3 = px.box(df, x="label_name", y=selected, color="label_name",
                      title=f"{selected} spread by class", color_discrete_map=color_map)
        st.plotly_chart(fig3, width='stretch', key=f'dataset_box_{selected}')

    c3, c4 = st.columns(2)
    with c3:
        x_feat = st.selectbox("Scatter X", numeric_cols, index=numeric_cols.index("URLSimilarityIndex"))
    with c4:
        y_feat = st.selectbox("Scatter Y", numeric_cols, index=numeric_cols.index("NoOfExternalRef"))
    fig4 = px.scatter(df.sample(min(3000, len(df)), random_state=1), x=x_feat, y=y_feat,
                       color="label_name", opacity=0.6, title=f"{x_feat} vs {y_feat}",
                       color_discrete_map=color_map)
    st.plotly_chart(fig4, width='stretch', key=f'dataset_scatter_{x_feat}_vs_{y_feat}')

    st.subheader("Correlation heatmap (top 15 most important features)")
    corr_df = pd.DataFrame(correlation["matrix"], index=correlation["features"], columns=correlation["features"])
    fig5 = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                      title="Feature correlation matrix")
    st.plotly_chart(fig5, width='stretch', key='dataset_correlation_heatmap')

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
        st.plotly_chart(fig_cm, width='stretch', key='model_confusion_matrix')
    with c2:
        fi = pd.Series(metrics["feature_importances"]).sort_values(ascending=True).tail(15)
        fig_fi = px.bar(x=fi.values, y=fi.index, orientation="h", title="Top 15 feature importances")
        st.plotly_chart(fig_fi, width='stretch', key='model_feature_importance')

    c3, c4 = st.columns(2)
    with c3:
        roc = metrics["roc_curve"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC curve"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                                      line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title=f"ROC curve (AUC = {metrics['roc_auc']:.4f})",
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, width='stretch', key='model_roc_curve')
    with c4:
        pr = metrics["pr_curve"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR curve"))
        fig_pr.update_layout(title="Precision-Recall curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig_pr, width='stretch', key='model_pr_curve')

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
        st.plotly_chart(fig_cm, width='stretch', key='nlp_confusion_matrix')
    with c2:
        lengths_df = pd.concat([
            pd.DataFrame({"length": title_lengths["legitimate"], "class": "legitimate"}),
            pd.DataFrame({"length": title_lengths["phishing"], "class": "phishing"}),
        ])
        fig_len = px.histogram(lengths_df, x="length", color="class", barmode="overlay", nbins=60,
                                range_x=[0, 100], title="Title length distribution by class",
                                color_discrete_map={"legitimate": "#2e7d32", "phishing": "#c62828"})
        st.plotly_chart(fig_len, width='stretch', key='nlp_title_length_distribution')

    st.subheader("Most predictive terms (TF-IDF + Logistic Regression coefficients)")
    c3, c4 = st.columns(2)
    with c3:
        legit_terms = pd.DataFrame(nlp_metrics["top_terms_legitimate"])
        fig_legit = px.bar(legit_terms.sort_values("weight"), x="weight", y="term", orientation="h",
                            title="Top terms → legitimate", color_discrete_sequence=["#2e7d32"])
        st.plotly_chart(fig_legit, width='stretch', key='nlp_legitimate_terms')
    with c4:
        phishing_terms = pd.DataFrame(nlp_metrics["top_terms_phishing"])
        phishing_terms["abs_weight"] = phishing_terms["weight"].abs()
        fig_phish = px.bar(phishing_terms.sort_values("abs_weight"), x="weight", y="term", orientation="h",
                            title="Top terms → phishing", color_discrete_sequence=["#c62828"])
        st.plotly_chart(fig_phish, width='stretch', key='nlp_phishing_terms')

    st.subheader("Most frequent words in titles")
    c5, c6 = st.columns(2)
    with c5:
        legit_words = pd.DataFrame(word_frequencies["legitimate"], columns=["word", "count"])
        fig_w1 = px.bar(legit_words.sort_values("count"), x="count", y="word", orientation="h",
                         title="Legitimate titles", color_discrete_sequence=["#2e7d32"])
        st.plotly_chart(fig_w1, width='stretch', key='nlp_legitimate_word_frequency')
    with c6:
        phishing_words = pd.DataFrame(word_frequencies["phishing"], columns=["word", "count"])
        fig_w2 = px.bar(phishing_words.sort_values("count"), x="count", y="word", orientation="h",
                         title="Phishing titles", color_discrete_sequence=["#c62828"])
        st.plotly_chart(fig_w2, width='stretch', key='nlp_phishing_word_frequency')

