# pyright: reportMissingImports=false

from dataclasses import asdict
import sys
from pathlib import Path
import json
import re
import time
from html import escape, unescape
from types import SimpleNamespace

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.display_helpers import (
    activity_time_label,
    format_gmail_message_label,
    friendly_model_name,
    render_explanation_sections,
    render_result_summary,
    render_signal_group,
)
from src.inference.baseline import BaselinePredictor
from src.inference.gmail_import import import_recent_gmail_messages, load_gmail_message_body
from src.explain.baseline_evidence import baseline_evidence
from src.explain.nlg import generate_explanation
from src.explain.rule_evidence import rule_based_evidence
from src.verdict_logic import compute_user_verdict, get_risk_level, result_advice

DATASET_STATS = ROOT / 'data' / 'processed' / 'dataset_stats.json'
BASELINE_DIR = ROOT / 'models' / 'baseline'
DISTILBERT_DIR = ROOT / 'models' / 'distilbert'
RESULTS_DIR = ROOT / 'results' / 'external'
GOOGLE_CREDENTIALS = ROOT / 'credentials.json'
GOOGLE_TOKEN = ROOT / 'token.json'
DISTILBERT_MODEL_DIR = DISTILBERT_DIR / 'final_model'
DISTILBERT_REQUIRED_FILES = (
    'config.json',
    'special_tokens_map.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'vocab.txt',
    'model.safetensors',
)

# Saved artefacts used by the Model/Results views.
BASELINE_METRICS = RESULTS_DIR / 'trec06_baseline_metrics.json'
DISTILBERT_METRICS = RESULTS_DIR / 'trec06_distilbert_metrics.json'
BASELINE_INTERNAL_METRICS = BASELINE_DIR / 'metrics.json'
DISTILBERT_INTERNAL_METRICS = DISTILBERT_DIR / 'metrics.json'
MODEL_COMPARISON_IMG = RESULTS_DIR / 'model_comparison.png'
BASELINE_INTERNAL_CM_IMG = BASELINE_DIR / 'confusion_matrix.png'
DISTILBERT_INTERNAL_CM_IMG = DISTILBERT_DIR / 'confusion_matrix.png'
BASELINE_CM_IMG = RESULTS_DIR / 'trec06_baseline_confusion_matrix.png'
DISTILBERT_CM_IMG = RESULTS_DIR / 'trec06_distilbert_confusion_matrix.png'
BASELINE_ROC_IMG = BASELINE_DIR / 'roc_curve.png'
BASELINE_FEAT_IMG = BASELINE_DIR / 'feature_importance.png'
SENSITIVITY_THRESHOLDS = {
    'Low risk': 0.77,
    'Balanced': 0.65,
    'Strict': 0.55,
}
APP_VERSION = '1.0.0'
RECENT_ACTIVITY_LIMIT = 5
RECENT_ACTIVITY_TTL = 24 * 60 * 60
SUPPORT_URL = 'https://github.com/ukashley/Veridexia'

# Remove hidden Unicode control characters from copied email text.
INVISIBLE_TRANSLATION = str.maketrans('', '', '\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff')


def load_json(path: Path):
    # Fail soft if a saved file is missing.
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def get_external_kpis(m: dict):
    if not m or 'report' not in m:
        return None

    rep = m['report']
    cls1 = rep.get('1', {})
    cls0 = rep.get('0', {})
    macro = rep.get('macro avg', {})
    weighted = rep.get('weighted avg', {})
    recall_phish = cls1.get('recall')
    recall_legit = cls0.get('recall')

    return {
        'accuracy': m.get('accuracy'),
        'f1_macro': macro.get('f1-score'),
        'f1_weighted': weighted.get('f1-score'),
        'recall_phish': recall_phish,
        'precision_phish': cls1.get('precision'),
        'f1_phish': cls1.get('f1-score'),
        'recall_legit': recall_legit,
        'false_negative_rate': (
            1.0 - float(recall_phish) if isinstance(recall_phish, (int, float)) else None
        ),
        'false_positive_rate': (
            1.0 - float(recall_legit) if isinstance(recall_legit, (int, float)) else None
        ),
    }


def get_internal_kpis(m: dict):
    if not m:
        return None

    test = m.get('test', {})
    cm = m.get('confusion_matrix') or []
    recall_phish = test.get('recall_phishing', test.get('recall'))
    recall_legit = test.get('recall_legitimate')
    precision_phish = test.get('precision_phishing', test.get('precision'))
    f1_phish = test.get('f1_phishing', test.get('f1'))
    false_positive_rate = None
    false_negative_rate = None

    if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
        tn, fp = cm[0]
        fn, tp = cm[1]
        legit_total = tn + fp
        phish_total = tp + fn
        if legit_total:
            recall_legit = tn / legit_total
            false_positive_rate = fp / legit_total
        if phish_total:
            recall_phish = tp / phish_total
            false_negative_rate = fn / phish_total

    if false_positive_rate is None and isinstance(recall_legit, (int, float)):
        false_positive_rate = 1.0 - float(recall_legit)
    if false_negative_rate is None and isinstance(recall_phish, (int, float)):
        false_negative_rate = 1.0 - float(recall_phish)

    return {
        'accuracy': test.get('accuracy'),
        'precision_phish': precision_phish,
        'recall_phish': recall_phish,
        'f1_phish': f1_phish,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
    }


def threshold_for_sensitivity(level: str) -> float:
    return float(SENSITIVITY_THRESHOLDS.get(level, SENSITIVITY_THRESHOLDS['Balanced']))


def distilbert_assets_available(model_dir: Path = DISTILBERT_MODEL_DIR) -> bool:
    return all((model_dir / filename).exists() for filename in DISTILBERT_REQUIRED_FILES)


def available_model_choices() -> list[str]:
    choices = ['baseline']
    if distilbert_assets_available():
        choices.insert(0, 'distilbert')
    return choices


def prune_recent_activity(entries: list[dict]) -> list[dict]:
    now = time.time()
    fresh = []
    for entry in entries or []:
        scanned_at = entry.get('scanned_at_ts')
        if isinstance(scanned_at, (int, float)) and now - float(scanned_at) <= RECENT_ACTIVITY_TTL:
            fresh.append(entry)
    return fresh[:RECENT_ACTIVITY_LIMIT]


def normalize_message_text(text: str) -> str:
    if not text:
        return ''
    # Gmail/exported email text can contain literal HTML entities such as &zwnj;.
    # Decode them before removing hidden characters so model input is less noisy.
    cleaned = unescape(str(text)).translate(INVISIBLE_TRANSLATION).replace('\u00a0', ' ')
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r' ?\n ?', '\n', cleaned)
    return cleaned.strip()


def sensitivity_for_threshold(threshold: float) -> str:
    try:
        threshold_value = float(threshold)
    except Exception:
        return 'Balanced'

    return min(
        SENSITIVITY_THRESHOLDS,
        key=lambda label: abs(threshold_value - SENSITIVITY_THRESHOLDS[label]),
    )


def build_model_input(body: str, subject: str = '', sender_email: str = '', sender_context: str = '') -> str:
    parts = []
    sender = normalize_message_text(sender_email)
    sender_display = normalize_message_text(sender_context)
    subj = normalize_message_text(subject)
    body_text = normalize_message_text(body)

    # The models were trained on text, so sender and subject are folded into the
    # same plain-text input rather than treated as separate model features.
    if sender_display and sender_display != sender:
        parts.append(f'Sender: {sender_display}')
    if sender:
        parts.append(f'From: {sender}')
    if subj:
        parts.append(f'Subject: {subj}')
    if body_text:
        parts.append(body_text)

    return '\n\n'.join(parts)


def useful_context_items(items):
    hidden_keys = {
        'sender_email_present',
        'email_address_present',
    }
    return [item for item in (items or []) if item.get('key') not in hidden_keys]


def inject_custom_css():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 34rem),
                radial-gradient(circle at 85% 8%, rgba(14, 165, 233, 0.09), transparent 30rem),
                #f7fbff;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        #MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 0.9rem;
            padding-bottom: 1.25rem;
        }
        .announcement-bar {
            background: linear-gradient(90deg, #0f5bd7, #0ea5e9);
            color: #ffffff;
            text-align: center;
            font-weight: 750;
            border-radius: 0 0 18px 18px;
            padding: 0.55rem 1rem;
            margin: -0.9rem 0 0.85rem 0;
            box-shadow: 0 14px 34px rgba(37, 99, 235, 0.18);
        }
        .site-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 22px;
            padding: 0.82rem 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 18px 44px rgba(15, 23, 42, 0.07);
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 0.78rem;
        }
        .brand-mark {
            width: 46px;
            height: 46px;
            border-radius: 15px;
            background: linear-gradient(135deg, #052e73, #1d74f5);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.05rem;
            font-weight: 850;
            letter-spacing: -0.04em;
            box-shadow: 0 10px 24px rgba(29, 116, 245, 0.27);
        }
        .brand-name {
            color: #0b2545;
            font-size: 1.1rem;
            font-weight: 850;
            line-height: 1.05;
        }
        .brand-subtitle {
            color: #2563eb;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 0.12rem;
        }
        .site-meta {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.55rem;
            color: #475569;
            font-size: 0.86rem;
            font-weight: 700;
            flex-wrap: wrap;
        }
        .site-meta span {
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: rgba(248, 250, 252, 0.86);
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
        }
        .hero-shell {
            position: relative;
            overflow: hidden;
            border-radius: 30px;
            padding: 2.35rem 2.15rem 1.65rem 2.15rem;
            background:
                linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(238, 247, 255, 0.97)),
                radial-gradient(circle at 88% 8%, rgba(37, 99, 235, 0.18), transparent 25rem);
            border: 1px solid rgba(148, 163, 184, 0.24);
            box-shadow: 0 26px 70px rgba(15, 23, 42, 0.09);
            margin-bottom: 1rem;
        }
        .hero-shell::after {
            content: "";
            position: absolute;
            right: -7rem;
            top: -8rem;
            width: 20rem;
            height: 20rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(37, 99, 235, 0.16), rgba(14, 165, 233, 0.08));
        }
        .hero-content {
            position: relative;
            z-index: 2;
            max-width: 850px;
            margin: 0 auto;
            text-align: center;
        }
        .hero-kicker {
            display: inline-flex;
            border-radius: 999px;
            padding: 0.45rem 0.82rem;
            background: #dffcf2;
            color: #047857;
            font-weight: 800;
            font-size: 0.84rem;
            margin-bottom: 0.75rem;
        }
        .hero-title {
            color: #07264f;
            font-size: clamp(2.15rem, 4vw, 4rem);
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: -0.055em;
            margin: 0;
        }
        .hero-copy {
            color: #4b617c;
            font-size: 1.05rem;
            line-height: 1.62;
            max-width: 760px;
            margin: 0.95rem auto 1.35rem auto;
        }
        .hero-actions {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1.35rem;
        }
        .primary-cta {
            display: inline-block;
            background: #1d74f5;
            color: #ffffff !important;
            text-decoration: none !important;
            font-weight: 850;
            min-width: min(430px, 82vw);
            text-align: center;
            padding: 1.05rem 3.3rem;
            border-radius: 16px;
            box-shadow: 0 16px 34px rgba(29, 116, 245, 0.25);
        }
        .primary-cta:hover {
            background: #0f5bd7;
            color: #ffffff !important;
            text-decoration: none !important;
        }
        .feature-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 0.95rem;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 0.86rem;
            text-align: left;
            min-height: 98px;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.045);
        }
        .feature-card.active {
            border-color: rgba(37, 99, 235, 0.45);
            box-shadow: 0 16px 34px rgba(37, 99, 235, 0.12);
        }
        .feature-card strong {
            color: #0b2545;
            display: block;
            font-size: 0.98rem;
            margin-bottom: 0.45rem;
        }
        .feature-card span {
            color: #64748b;
            font-size: 0.86rem;
            line-height: 1.55;
        }
        div[data-baseweb="tab-list"] {
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            background: rgba(255, 255, 255, 0.80);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            padding: 0.34rem;
            margin-bottom: 0.7rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.045);
        }
        button[data-baseweb="tab"] {
            width: 100%;
            border-radius: 12px !important;
            font-weight: 750;
            justify-content: center;
            padding: 0.72rem 0.85rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: #eff6ff;
            color: #1d4ed8;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #2563eb !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] * {
            color: #1d4ed8 !important;
        }
        div[data-testid="stToggle"] button[role="switch"][aria-checked="true"],
        div[data-testid="stToggle"] div[role="switch"][aria-checked="true"],
        div[data-testid="stToggle"] input:checked + div {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
        }
        div[data-testid="stToggle"] button[role="switch"]:focus,
        div[data-testid="stToggle"] div[role="switch"]:focus {
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.16) !important;
        }
        div[data-testid="stToggle"] [style*="rgb(255, 75, 75)"],
        div[data-testid="stSlider"] [style*="rgb(255, 75, 75)"],
        div[data-testid="stToggle"] [style*="#ff4b4b"],
        div[data-testid="stSlider"] [style*="#ff4b4b"] {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
            color: #2563eb !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background-color: #2563eb !important;
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }
        .result-card {
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.9rem;
            background: rgba(235, 245, 255, 0.55);
        }
        .result-label {
            font-size: 1.55rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .result-meta {
            font-size: 0.95rem;
            color: #4a5568;
        }
        .explanation-card {
            border-left: 5px solid #2563eb;
            background: rgba(239, 246, 255, 0.8);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .page-heading {
            font-size: 2.15rem;
            font-weight: 700;
            line-height: 1.15;
            margin: 1.05rem 0 0.55rem 0;
        }
        div.stButton > button[kind="primary"] {
            background-color: #2563eb;
            border-color: #2563eb;
            color: white;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1d4ed8;
            border-color: #1d4ed8;
            color: white;
        }
        .small-note {
            color: #6b7280;
            font-size: 0.92rem;
        }
        .empty-results-card {
            background: linear-gradient(135deg, rgba(239, 246, 255, 0.95), rgba(255, 255, 255, 0.96));
            border: 1px solid rgba(96, 165, 250, 0.28);
            border-radius: 18px;
            padding: 1.05rem 1.15rem;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.055);
        }
        .empty-results-title {
            color: #0b2545;
            font-weight: 850;
            font-size: 1.05rem;
            margin-bottom: 0.35rem;
        }
        .empty-results-text {
            color: #4b617c;
            line-height: 1.55;
        }
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid rgba(100, 116, 139, 0.35) !important;
            box-shadow: inset 0 1px 0 rgba(15, 23, 42, 0.025);
        }
        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="textarea"] textarea:focus,
        div[data-baseweb="select"] > div:focus-within {
            border-color: rgba(37, 99, 235, 0.58) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.10);
        }
        .big-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 22px;
        }
        .big-table th {
            text-align: left;
            padding: 10px 12px;
            font-size: 24px;
            font-weight: 700;
            border-bottom: 2px solid #ccc;
        }
        .big-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 3rem;
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.14), transparent 18rem),
                linear-gradient(180deg, rgba(239, 246, 255, 0.96), rgba(247, 251, 255, 0.96));
            border-right: 1px solid rgba(96, 165, 250, 0.24);
            box-shadow: 10px 0 30px rgba(15, 23, 42, 0.045);
        }
        section[data-testid="stSidebar"] {
            width: 268px !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #0b2545;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(96, 165, 250, 0.26);
        }
        .site-footer {
            margin-top: 1.35rem;
            background: #111827;
            color: #e5e7eb;
            border-radius: 26px 26px 0 0;
            padding: 1.45rem 1.6rem;
        }
        .footer-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr 1fr 1fr;
                gap: 1.2rem;
        }
        .footer-title {
            color: #ffffff;
            font-size: 1.08rem;
            font-weight: 850;
            margin-bottom: 0.65rem;
        }
        .footer-text, .footer-link {
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.65;
            margin-bottom: 0.38rem;
        }
        .footer-link {
            display: block;
            text-decoration: none;
        }
        @media (max-width: 900px) {
            .site-header {
                align-items: flex-start;
                flex-direction: column;
            }
            .site-meta {
                justify-content: flex-start;
            }
            .hero-shell {
                padding: 2.2rem 1.25rem 1.35rem 1.25rem;
            }
            .feature-row, .footer-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_heading(title: str):
    slug = title.lower().replace(' ', '-')
    st.markdown(f"<h1 id='{escape(slug)}' class='page-heading'>{escape(title)}</h1>", unsafe_allow_html=True)


def render_site_header():
    # Header and feature chips make the Streamlit prototype feel more like a small web app.
    st.markdown(
        """
        <div class="announcement-bar">
            Explainable phishing email detection application for safer message review
        </div>
        <div class="site-header">
            <div class="brand-lockup">
                <div class="brand-mark">VX</div>
                <div>
                    <div class="brand-name">Veridexia</div>
                    <div class="brand-subtitle">ML Phishing Detection</div>
                </div>
            </div>
            <div class="site-meta">
                <span>Email scanner</span>
                <span>Gmail import</span>
                <span>Evidence explanations</span>
                <span>Decision support</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    # Landing hero summarises the app before the user reaches the scan form.
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-content">
                <div class="hero-kicker">Explainable email threat checking</div>
                <h1 class="hero-title">Scan emails for phishing signals before you click.</h1>
                <p class="hero-copy">
                    Veridexia analyzes sender, subject and email content with a DistilBERT classifier
                    and evidence-supported signals, then explains why a message looks safe, suspicious or high risk.
                </p>
                <div class="hero-actions">
                    <a class="primary-cta" href="#scan-email">Start scanning</a>
                </div>
                <div class="feature-row">
                    <div class="feature-card">
                        <strong>Email scanner</strong>
                        <span>Paste email text or import messages locally from Gmail for analysis.</span>
                    </div>
                    <div class="feature-card">
                        <strong>ML classification</strong>
                        <span>DistilBERT is the main model, with a baseline available in advanced mode.</span>
                    </div>
                    <div class="feature-card">
                        <strong>Evidence layer</strong>
                        <span>Highlights urgency, credential requests, links, sender patterns, and reassurance cues.</span>
                    </div>
                    <div class="feature-card">
                        <strong>Decision support</strong>
                        <span>Designed to support user judgement, not replace careful review.</span>
                    </div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_site_footer():
    # Footer reinforces the safety framing: this is decision support, not a guaranteed defence.
    st.markdown(
        """
        <footer class="site-footer">
            <div class="footer-grid">
                <div>
                    <div class="footer-title">Veridexia</div>
                    <div class="footer-text">
                        A web application for explainable phishing email detection.
                        Results should be treated as decision support rather than guaranteed protection.
                    </div>
                </div>
                <div>
                    <div class="footer-title">Features</div>
                    <span class="footer-link">Email scanning</span>
                    <span class="footer-link">Gmail import</span>
                    <span class="footer-link">Evidence explanations</span>
                </div>
                <div>
                    <div class="footer-title">Technology</div>
                    <span class="footer-link">Python and Streamlit</span>
                    <span class="footer-link">DistilBERT and baseline ML</span>
                    <span class="footer-link">Risk signal detection</span>
                </div>
                <div>
                    <div class="footer-title">Safety</div>
                    <span class="footer-link">Do not share passwords</span>
                    <span class="footer-link">Verify through official websites</span>
                    <span class="footer-link">Review uncertain results manually</span>
                </div>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_baseline_predictor():
    # Load once per session so repeated scans are fast.
    return BaselinePredictor()


@st.cache_resource(show_spinner=False)
def get_distilbert_predictor():
    # DistilBERT is heavier, so caching avoids reloading the transformer every scan.
    if not distilbert_assets_available():
        raise RuntimeError('DistilBERT weights are not available in this local copy.')
    from src.inference.distilbert import DistilBertPredictor
    return DistilBertPredictor()


st.set_page_config(page_title='Veridexia', layout='wide')
inject_custom_css()
render_site_header()
render_hero()

# Available runtime features depend on local model and OAuth files.
model_choices = available_model_choices()
distilbert_enabled = 'distilbert' in model_choices
gmail_enabled = GOOGLE_CREDENTIALS.exists()

# Session state keeps the form, Gmail previews and recent scans alive across Streamlit reruns.
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = 'distilbert' if distilbert_enabled else 'baseline'
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.65
if 'detection_sensitivity' not in st.session_state:
    st.session_state.detection_sensitivity = sensitivity_for_threshold(st.session_state.threshold)
if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'scan_sender_email' not in st.session_state:
    st.session_state.scan_sender_email = ''
if 'scan_subject' not in st.session_state:
    st.session_state.scan_subject = ''
if 'scan_email_text' not in st.session_state:
    st.session_state.scan_email_text = ''
if 'scan_sender_context' not in st.session_state:
    st.session_state.scan_sender_context = ''
if 'gmail_messages' not in st.session_state:
    st.session_state.gmail_messages = []
if 'gmail_import_warnings' not in st.session_state:
    st.session_state.gmail_import_warnings = []
if 'gmail_import_status' not in st.session_state:
    st.session_state.gmail_import_status = ''
if 'gmail_import_status_type' not in st.session_state:
    st.session_state.gmail_import_status_type = 'info'
if 'gmail_selected_index' not in st.session_state:
    st.session_state.gmail_selected_index = 0
if 'gmail_max_results' not in st.session_state:
    st.session_state.gmail_max_results = 5
if 'recent_activity' not in st.session_state:
    st.session_state.recent_activity = []

st.session_state.recent_activity = prune_recent_activity(st.session_state.recent_activity)
if st.session_state.last_result:
    scanned_at = st.session_state.last_result.get('scanned_at_ts')
    if isinstance(scanned_at, (int, float)) and time.time() - float(scanned_at) > RECENT_ACTIVITY_TTL:
        st.session_state.last_result = None

if st.session_state.model_choice not in model_choices:
    st.session_state.model_choice = model_choices[0]

with st.sidebar:
    # Global controls. Baseline is kept in Advanced mode because DistilBERT is the main model.
    st.markdown('# Settings')
    st.session_state.advanced_mode = st.toggle(
        'Advanced mode',
        value=st.session_state.advanced_mode,
        help='Show advanced controls such as model choice and detection sensitivity.',
    )

    if st.session_state.advanced_mode:
        st.session_state.model_choice = st.selectbox(
            'Model',
            model_choices,
            index=model_choices.index(st.session_state.model_choice),
            format_func=friendly_model_name,
            help="Choose which classifier makes the prediction. DistilBERT is the project's main transformer-based model, while the baseline remains available for comparison.",
        )
    else:
        st.session_state.model_choice = 'distilbert' if distilbert_enabled else 'baseline'

    if not distilbert_enabled:
        st.caption('This local copy is using the baseline model because the DistilBERT weights are not available.')

    if st.session_state.advanced_mode:
        st.markdown('#### Detection sensitivity')
        st.session_state.detection_sensitivity = st.select_slider(
            'Detection sensitivity',
            options=list(SENSITIVITY_THRESHOLDS.keys()),
            value=st.session_state.detection_sensitivity,
            label_visibility='collapsed',
            help='Low risk is stricter and reduces false alarms. Strict catches more suspicious emails but may flag more legitimate messages.',
        )
        st.caption(
            f"Current phishing threshold: {threshold_for_sensitivity(st.session_state.detection_sensitivity):.2f}"
        )

    st.session_state.threshold = threshold_for_sensitivity(st.session_state.detection_sensitivity)

    st.divider()
    st.markdown('#### Current setup')
    st.caption(f"Model: {friendly_model_name(st.session_state.model_choice)}")
    st.caption(f"Sensitivity: {st.session_state.detection_sensitivity}")
    st.caption('Detailed evidence appears inside the result expanders after each scan.')


st.markdown("<div id='app-tabs'></div>", unsafe_allow_html=True)

# Main app tabs
tab_scan, tab_activity, tab_about, tab_guide = st.tabs(
    ['Scan Email', 'Recent Activity', 'About', 'Guide']
)

with tab_scan:
    # Main scan workflow to collect input, run the selected model, then explain the result.
    render_page_heading('Scan Email')
    st.caption('Paste the body of an email or import a recent Gmail message. The classifier uses the email content together with the sender email and subject when available.')

    left, right = st.columns([1.15, 1])

    with left:
        # Input panel
        st.markdown('#### Gmail import')
        if gmail_enabled:
            st.caption('Import recent emails from your Gmail inbox to fill the form automatically. The first time you use this locally, Google will ask you to sign in.')
            st.markdown('**Recent inbox emails**')
            gmail_col1, gmail_col2 = st.columns([1, 1])
            with gmail_col1:
                st.selectbox(
                    'Recent inbox emails',
                    options=[5, 10, 20],
                    key='gmail_max_results',
                    label_visibility='collapsed',
                    help='Choose how many recent inbox messages to fetch from Gmail.',
                )
            with gmail_col2:
                import_gmail = st.button('Import recent emails', type='primary', use_container_width=True)

            if import_gmail:
                # First import only fetches previews/metadata so the app stays responsive.
                with st.spinner('Importing recent Gmail previews...'):
                    gmail_result = import_recent_gmail_messages(
                        GOOGLE_CREDENTIALS,
                        GOOGLE_TOKEN,
                        max_results=int(st.session_state.gmail_max_results),
                    )
                st.session_state.gmail_messages = [asdict(item) for item in gmail_result.items]
                st.session_state.gmail_import_warnings = gmail_result.warnings
                if gmail_result.items:
                    st.session_state.gmail_selected_index = 0
                    first_item = st.session_state.gmail_messages[0]
                    st.session_state.scan_sender_email = first_item.get('sender_email', '')
                    st.session_state.scan_sender_context = first_item.get('sender_label') or first_item.get('sender_email', '')
                    st.session_state.scan_subject = first_item.get('subject', '')
                    st.session_state.scan_email_text = first_item.get('snippet', '')
                    st.session_state.gmail_import_status = (
                        f"Imported {len(gmail_result.items)} inbox email(s). "
                        'The first message preview has been loaded into the scan form.'
                    )
                    st.session_state.gmail_import_status_type = 'success'
                else:
                    if gmail_result.warnings:
                        st.session_state.gmail_import_status = 'Gmail import could not be completed.'
                        st.session_state.gmail_import_status_type = 'warning'
                    else:
                        st.session_state.gmail_import_status = 'No Gmail messages were imported.'
                        st.session_state.gmail_import_status_type = 'info'

            if st.session_state.gmail_import_status:
                status_type = st.session_state.gmail_import_status_type
                if status_type == 'success':
                    st.success(st.session_state.gmail_import_status)
                elif status_type == 'warning':
                    st.warning(st.session_state.gmail_import_status)
                else:
                    st.info(st.session_state.gmail_import_status)

            for warning in st.session_state.gmail_import_warnings:
                st.warning(warning)
        else:
            st.caption('Gmail import is not enabled because local Google OAuth credentials were not found.')
            import_gmail = False

        if gmail_enabled and st.session_state.gmail_messages:
            st.markdown('**Imported Gmail messages**')
            st.selectbox(
                'Imported Gmail messages',
                options=list(range(len(st.session_state.gmail_messages))),
                index=min(st.session_state.gmail_selected_index, len(st.session_state.gmail_messages) - 1),
                format_func=lambda idx: format_gmail_message_label(st.session_state.gmail_messages[idx]),
                key='gmail_selected_index',
                label_visibility='collapsed',
            )
            if st.button('Load selected email', type='primary', use_container_width=True):
                selected_message = st.session_state.gmail_messages[st.session_state.gmail_selected_index]
                if not selected_message.get('body_loaded'):
                    # Load the full body only when the user chooses a message to analyse.
                    with st.spinner('Loading selected email...'):
                        full_message, load_warnings = load_gmail_message_body(
                            GOOGLE_CREDENTIALS,
                            GOOGLE_TOKEN,
                            selected_message.get('message_id', ''),
                        )
                    st.session_state.gmail_import_warnings.extend(load_warnings)
                    if full_message is not None:
                        selected_message.update(asdict(full_message))
                        st.session_state.gmail_messages[st.session_state.gmail_selected_index] = selected_message
                        st.session_state.gmail_import_status = 'The selected email has been fully loaded.'
                        st.session_state.gmail_import_status_type = 'success'

                st.session_state.scan_sender_email = selected_message.get('sender_email', '')
                st.session_state.scan_sender_context = selected_message.get('sender_label') or selected_message.get('sender_email', '')
                st.session_state.scan_subject = selected_message.get('subject', '')
                st.session_state.scan_email_text = selected_message.get('body_text') or selected_message.get('snippet', '')
                st.rerun()

        st.markdown('**Sender email**')
        sender_email = st.text_input('Sender email', key='scan_sender_email', label_visibility='collapsed')
        st.markdown('**Subject**')
        subject_line = st.text_input('Subject', key='scan_subject', label_visibility='collapsed')
        st.markdown('**Email Content**')
        email_text = st.text_area('Email Content', height=220, key='scan_email_text', label_visibility='collapsed')

        run = st.button('Analyze', type='primary', use_container_width=True)

    with right:
        # Output panel
        st.subheader('Results')

        if run:
            # Normalise and combine all visible user input before model inference.
            model_choice = st.session_state.model_choice
            threshold = float(st.session_state.threshold)
            effective_sender = (sender_email or '').strip()
            effective_sender_context = (
                st.session_state.scan_sender_context.strip()
                if isinstance(st.session_state.scan_sender_context, str)
                else ''
            )
            if effective_sender and effective_sender_context and effective_sender not in effective_sender_context:
                effective_sender_context = f'{effective_sender_context} <{effective_sender}>'
            effective_subject = (subject_line or '').strip()
            combined_body = email_text.strip()
            combined_body = normalize_message_text(combined_body)
            effective_sender = normalize_message_text(effective_sender)
            effective_subject = normalize_message_text(effective_subject)
            model_input = build_model_input(
                combined_body,
                subject=effective_subject,
                sender_email=effective_sender,
                sender_context=effective_sender_context,
            )
            if not model_input.strip():
                st.warning('Paste some email content or import a Gmail message first.')
                st.stop()

            with st.spinner(f'Analysing email with {friendly_model_name(model_choice)}...'):
                start = time.time()

                if model_choice == 'baseline':
                    # Baseline path: fast TF-IDF + Logistic Regression prediction.
                    predictor = get_baseline_predictor()
                    result = predictor.predict(model_input, threshold=threshold)
                    evidence = baseline_evidence(model_input, predictor)
                    support_result = None
                else:
                    # DistilBERT path: main transformer prediction plus baseline as a support check.
                    predictor = get_distilbert_predictor()
                    result = predictor.predict(model_input, threshold=threshold)
                    evidence = None
                    support_predictor = get_baseline_predictor()
                    support_result = support_predictor.predict(model_input, threshold=threshold)

                rules = rule_based_evidence(combined_body, sender_email=effective_sender, subject=effective_subject)
                verdict = compute_user_verdict(result, rules, support_result=support_result)
                p_phish = float(verdict.get('display_prob', result.prob_phishing))
                ms = (time.time() - start) * 1000.0
                scanned_at = time.time()
                explanation_result = SimpleNamespace(
                    label=1 if verdict['label'] == 'Phishing' else 0,
                    prob_phishing=p_phish,
                    threshold=result.threshold,
                )
                summary = generate_explanation(
                    explanation_result,
                    evidence=evidence,
                    rule_evidence=rules,
                    display_label=verdict['label'],
                    display_prob=p_phish,
                )

            risk_items = rules.get('signals', [])
            reassurance_items = rules.get('reassurance_signals', [])
            context_items = useful_context_items(rules.get('context_signals', []))
            risk_level = get_risk_level(
                verdict['label'],
                p_phish,
                result.threshold,
                bool(verdict['review_recommended']),
            )
            advice = result_advice(verdict['label'], risk_level, p_phish, result.threshold)

            render_result_summary(verdict, risk_level, result.model_name, advice, p_phish)

            render_explanation_sections(
                summary=summary,
                rules=rules,
                risk_items=risk_items,
                reassurance_items=reassurance_items,
                context_items=context_items,
                expanded=False,
            )

            history_entry = {
                # Store only the small result snapshot needed for Recent Activity.
                'model': result.model_name,
                'threshold': float(threshold),
                'display_verdict': verdict['label'],
                'risk_level': risk_level,
                'advice': advice,
                'phishing_probability': float(p_phish),
                'risk_score': float(rules.get('risk_score', 0.0)),
                'inference_ms': float(ms),
                'review_recommended': bool(verdict['review_recommended']),
                'sender_email': effective_sender,
                'subject': effective_subject,
                'scanned_at_ts': scanned_at,
                'explanation_summary': summary,
                'warning_signals': risk_items,
                'reassurance_signals': reassurance_items,
                'context_signals': context_items,
            }
            st.session_state.last_result = history_entry
            st.session_state.recent_activity = prune_recent_activity(
                [history_entry, *st.session_state.recent_activity]
            )
        else:
            st.markdown(
                """
                <div class="empty-results-card">
                    <div class="empty-results-title">Ready to scan</div>
                    <div class="empty-results-text">
                        Scan results will appear here, including the classification, risk level,
                        confidence score, and explanation.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

with tab_activity:
    render_page_heading('Recent Activity')
    st.caption('Recent activity is kept only for this session. It can be cleared manually and expires after 24 hours.')

    if st.button('Clear History'):
        st.session_state.recent_activity = []
        st.session_state.last_result = None
        st.rerun()

    history = prune_recent_activity(st.session_state.recent_activity)
    st.session_state.recent_activity = history

    if not history:
        st.info('No emails have been scanned yet. Use the Scan Email tab to analyze a message.')
    else:
        for item in history:
            # Recent Activity is session-only; it is not written to a database.
            verdict = item.get('display_verdict', 'Unknown')
            sender = item.get('sender_email') or 'Unknown sender'
            subject = item.get('subject') or '(no subject)'
            model_label = friendly_model_name(item.get('model', ''))
            scanned_label = activity_time_label(item.get('scanned_at_ts'))
            summary = item.get('explanation_summary', '')
            warning_signals = item.get('warning_signals') or []
            reassurance_signals = item.get('reassurance_signals') or []
            context_signals = item.get('context_signals') or []

            with st.container(border=True):
                top_left, top_right = st.columns([3, 1.3])
                with top_left:
                    st.markdown(f"**Subject:** {subject}")
                    st.markdown(f"**Sender:** {sender}")
                    st.caption(f'Scanned: {scanned_label}')
                with top_right:
                    st.markdown(f"**Verdict:** {verdict}")
                    if item.get('review_recommended'):
                        st.caption('Review recommended')

                meta1, meta2 = st.columns(2)
                meta1.caption(f'Model: {model_label}')
                meta2.caption(f'Saved: {scanned_label}')

                with st.expander('View saved explanation'):
                    if summary:
                        safe_summary = escape(summary).replace('\n', '<br>')
                        st.markdown(
                            f"<div class='explanation-card'>{safe_summary}</div>",
                            unsafe_allow_html=True,
                        )

                        if warning_signals or reassurance_signals or context_signals:
                            render_signal_group('Warning signs', warning_signals, '!')
                            render_signal_group('Additional context', context_signals, '')
                    else:
                        st.info('This scan was saved before explanation snapshots were added to Recent Activity.')

with tab_about:
    render_page_heading('About')
    st.write('Veridexia is a phishing email detection application that combines ML classification with evidence-supported explanations.')

    st.markdown('### What Veridexia does')
    st.markdown(
        """
        Veridexia analyses email text and estimates whether it is likely to be phishing or legitimate.
        It also highlights suspicious signals such as urgency, credential requests, links, and impersonation patterns.
        """
    )

    st.markdown('### Privacy and data handling')
    st.markdown(
        """
        - This application does not write scanned emails to a database.
        - Recent Activity is kept only in the current session and can be cleared at any time.
        - The session history is automatically cleared after 24 hours.
        - Email content is processed only to generate the current classification and explanation.
        """
    )

    st.markdown('### Limitations and disclaimer')
    st.markdown(
        """
        Veridexia is a decision-support tool. It may produce false positives or false negatives,
        so results should be reviewed alongside the email context and user judgement.
        """
    )

    st.markdown('### Google sign-in')
    st.markdown(
        """
        If Gmail import is used, sign-in is handled through Google OAuth.
        This app never sees or stores your Google password directly.
        """
    )

    st.markdown('### Version')
    st.code(APP_VERSION)

    st.markdown('### Support')
    st.markdown(f'[Project page / support]({SUPPORT_URL})')

with tab_guide:
    render_page_heading('Guide')
    st.write('This page explains how to get cleaner scan results and how Gmail import works.')

    st.divider()
    st.markdown('### For best results')
    st.info(
        'Include the sender, subject, and full email body where possible. '
        'Short extracts may remove useful context and make the scan less reliable.'
    )

    st.divider()
    st.markdown('### Supported inputs')
    st.markdown(
        """
        You can paste email text directly or import recent messages from Gmail. For the clearest result,
        include the sender, subject, and full email body where possible.
        """
    )

    st.divider()
    st.markdown('### Understanding results')
    st.markdown(
        """
        Always read the explanation as well as the final phishing or legitimate label.
        Some legitimate emails, such as newsletters, job alerts, and account notices, may still trigger warning signs.
        The system supports decision-making but cannot guarantee that every phishing email will be detected.
        """
    )
    st.info('Review recommended means the email contains signals that should be checked manually.')

    st.divider()
    st.markdown('### Gmail import and OAuth')
    st.markdown(
        """
        Gmail import uses Google OAuth sign-in. Google handles the sign-in screen directly,
        so Veridexia does not see or store your Google password.
        """
    )

    st.divider()
    st.markdown('### Safety note')
    st.warning('Do not click links or share account details until the message has been checked through an official website or trusted contact channel.')

render_site_footer()