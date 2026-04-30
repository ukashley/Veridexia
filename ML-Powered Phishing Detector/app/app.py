# pyright: reportMissingImports=false

from dataclasses import asdict
import sys
from pathlib import Path
import json
import re
import time
from html import escape
from types import SimpleNamespace

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.baseline import BaselinePredictor
from src.inference.gmail_import import import_recent_gmail_messages, load_gmail_message_body
from src.inference.upload_extractors import build_upload_context
from src.explain.baseline_evidence import baseline_evidence
from src.explain.nlg import generate_explanation
from src.explain.rule_evidence import rule_based_evidence

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

# Saved artefacts
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


def fmt_pct(x):
    try:
        return f'{float(x) * 100:.2f}%'
    except Exception:
        return '-'


def fmt_count(x):
    try:
        return f'{int(x):,}'
    except Exception:
        return '-'


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


def render_metrics_table(rows: list[dict]):
    if not rows:
        return

    df = pd.DataFrame(rows)
    for col in [
        'Accuracy',
        'Precision (phish)',
        'Recall (phish)',
        'F1 (phish)',
        'False positive rate',
        'False negative rate',
    ]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt_pct(x) if isinstance(x, (int, float)) else '-')
    html = df.to_html(index=False)
    st.markdown(f"<div class='big-table'>{html}</div>", unsafe_allow_html=True)


def severity_color(level: str):
    return {
        'high': 'High risk',
        'medium': 'Moderate risk',
        'review': 'Needs review',
        'low': 'Low risk',
    }.get(level, 'Needs review')


def format_gmail_message_label(item: dict) -> str:
    subject = (item.get('subject') or '(no subject)').strip()
    sender = (item.get('sender_email') or item.get('sender_label') or 'unknown sender').strip()
    received = (item.get('received_at') or '').strip()
    if received:
        return f'{subject} - {sender} ({received})'
    return f'{subject} - {sender}'


def friendly_model_name(model_name: str) -> str:
    return {
        'distilbert': 'DistilBERT',
        'baseline': 'Baseline',
        'baseline_tfidf_lr': 'Baseline',
    }.get((model_name or '').strip().lower(), model_name or 'Unknown')


def confidence_label(prob: float, threshold: float):
    gap = abs(float(prob) - float(threshold))
    if gap >= 0.25:
        return 'High'
    if gap >= 0.12:
        return 'Moderate'
    return 'Mixed'


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


def activity_time_label(scanned_at: float | None) -> str:
    if not isinstance(scanned_at, (int, float)):
        return 'Unknown time'
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(float(scanned_at)))


def normalize_message_text(text: str) -> str:
    if not text:
        return ''
    cleaned = str(text).translate(INVISIBLE_TRANSLATION).replace('\u00a0', ' ')
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


def compute_user_verdict(result, rules: dict, support_result=None):
    prob = float(result.prob_phishing)
    threshold = float(result.threshold)
    risk_items = rules.get('signals', [])
    reassurance_items = rules.get('reassurance_signals', [])
    context_items = rules.get('context_signals', [])
    risk_keys = {item.get('key') for item in risk_items}
    reassurance_keys = {item.get('key') for item in reassurance_items}
    context_keys = {item.get('key') for item in context_items}
    risk_score = float(rules.get('risk_score', 0.0))
    support_prob = None if support_result is None else float(support_result.prob_phishing)
    risk_count = len(risk_items)

    no_strong_cues = (
        not bool(rules.get("has_credential_request"))
        and not bool(rules.get("has_threat"))
        and not bool(rules.get("has_urgency"))
        and int(rules.get("url_count", 0)) == 0
    )

    # High-risk cues
    dangerous_risk_keys = {
        'credential_request',
        'threat_language',
        'suspicious_link',
        'domain_mismatch',
        'suspicious_sender_domain',
    }
    no_dangerous_cues = not bool(dangerous_risk_keys & risk_keys)
    has_model_support = support_prob is not None
    max_model_prob = max(prob, support_prob if support_prob is not None else 0.0)
    strong_phishing_combo = (
        (
            'credential_request' in risk_keys
            and (
                bool({'urgency', 'threat_language', 'suspicious_link', 'domain_mismatch', 'suspicious_sender_domain'} & risk_keys)
                or max_model_prob >= 0.55
            )
        )
        or (
            'suspicious_link' in risk_keys
            and (
                bool({'domain_mismatch', 'suspicious_sender_domain'} & risk_keys)
                or max_model_prob >= threshold
            )
        )
        or (
            risk_score >= 3.0
            and max_model_prob >= 0.45
        )
    )

    near_boundary = abs(prob - threshold) < 0.12
    review_recommended = near_boundary or (prob >= threshold and (no_strong_cues or no_dangerous_cues))
    benign_context = (
        'sender_email_present' in context_keys
        and 'email_address_present' in context_keys
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    )
    clean_message = no_strong_cues and risk_score <= 0 and 'domain_mismatch' not in risk_keys and 'suspicious_link' not in risk_keys
    low_severity_risk_keys = {'payment_language', 'generic_greeting'}
    low_severity_only = bool(risk_keys) and risk_keys.issubset(low_severity_risk_keys)

    # Routine categories
    routine_message_keys = {
        'transactional_notification',
        'account_administration_notice',
        'newsletter_context',
        'job_alert_context',
        'identity_verification_context',
        'formal_service_message',
    }
    trusted_service_keys = {
        'sender_brand_match',
        'domain_match',
        'link_domain_match',
        'official_website_present',
        'support_contact_present',
        'no_email_credential_request',
        'security_notification',
    }
    transactional_reassurance = bool(
        (routine_message_keys | trusted_service_keys) & reassurance_keys
    )
    strong_benign_reassurance = bool(
        routine_message_keys & reassurance_keys
    ) and bool({'sender_brand_match', 'domain_match', 'link_domain_match'} & reassurance_keys)
    trusted_service_context = (
        no_dangerous_cues
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
        and bool({'sender_brand_match', 'domain_match', 'link_domain_match'} & reassurance_keys)
        and bool({'transactional_notification', 'formal_service_message', 'support_contact_present', 'official_website_present'} & reassurance_keys)
        and bool({'no_email_credential_request', 'security_notification'} & reassurance_keys)
        and risk_score <= 1.2
    )
    safe_routine_message = (
        bool(routine_message_keys & reassurance_keys)
        and no_dangerous_cues
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
        and risk_score <= 0
    )
    benign_model_disagreement = (
        prob >= threshold
        and has_model_support
        and support_prob <= 0.45
        and no_dangerous_cues
        and risk_score <= 0.5
        and (
            bool(routine_message_keys & reassurance_keys)
            or bool({'sender_brand_match', 'domain_match', 'link_domain_match', 'security_notification', 'identity_verification_context'} & reassurance_keys)
            or benign_context
            or not risk_keys
        )
    )
    display_prob = prob
    decision_basis = 'model'

    # Manual overrides
    if strong_phishing_combo:
        main_label = "Phishing"
        level = "high" if max_model_prob >= 0.75 or risk_score >= 3.4 else "medium"
        display_prob = max(max_model_prob, threshold + 0.05)
        decision_basis = 'high_risk_evidence_override'
    elif (
        prob >= threshold
        and 'identity_verification_context' in reassurance_keys
        and ('sender_email_present' in context_keys or bool({'sender_brand_match', 'domain_match'} & reassurance_keys))
        and no_dangerous_cues
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(
            support_prob if support_prob is not None else threshold - 0.01,
            max(0.0, threshold - 0.01),
        )
        review_recommended = False
        decision_basis = 'identity_verification_override'
    elif (
        prob >= threshold
        and trusted_service_context
        and (support_prob is None or support_prob <= 0.90)
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(
            support_prob if support_prob is not None else threshold - 0.01,
            max(0.0, threshold - 0.01),
        )
        review_recommended = True
        decision_basis = 'trusted_service_context_override'
    elif (
        prob >= threshold
        and safe_routine_message
        and bool({'sender_brand_match', 'domain_match', 'link_domain_match'} & reassurance_keys)
        and (support_prob is None or support_prob <= 0.85)
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(
            support_prob if support_prob is not None else threshold - 0.01,
            max(0.0, threshold - 0.01),
        )
        decision_basis = 'benign_routine_override'
    elif benign_model_disagreement:
        main_label = "Legitimate"
        level = "review"
        display_prob = min(support_prob, max(0.0, threshold - 0.01))
        decision_basis = 'benign_support_model_override'
    elif prob >= threshold and near_boundary and no_dangerous_cues and risk_score <= 0 and (
        benign_context or bool({'domain_match', 'link_domain_match'} & reassurance_keys)
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        decision_basis = 'hybrid_override'
    elif (
        prob >= threshold
        and support_prob is not None
        and support_prob < threshold
        and no_dangerous_cues
        and low_severity_only
        and risk_count <= 1
        and transactional_reassurance
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = support_prob
        decision_basis = 'transactional_override'
    elif (
        prob >= threshold
        and support_prob is not None
        and (clean_message or safe_routine_message)
        and ('sender_email_present' in context_keys or benign_context or bool({'domain_match', 'link_domain_match'} & reassurance_keys))
        and support_prob <= 0.40
    ):
        ensemble_prob = round((prob + support_prob) / 2.0, 4)
        if ensemble_prob < threshold:
            main_label = "Legitimate"
            level = "review"
            display_prob = ensemble_prob
            decision_basis = 'cross_model_override'
        else:
            main_label = "Phishing"
            level = "medium"
    elif (
        prob >= threshold
        and no_dangerous_cues
        and risk_score <= -1.0
        and strong_benign_reassurance
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        decision_basis = 'benign_notice_override'
    elif (
        prob >= threshold
        and support_prob is not None
        and no_dangerous_cues
        and risk_score <= -0.6
        and bool(routine_message_keys & reassurance_keys)
        and support_prob <= max(0.55, threshold - 0.05)
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = min(support_prob, max(0.0, threshold - 0.01))
        decision_basis = 'routine_message_override'
    elif prob >= threshold:
        main_label = "Phishing"
        level = "high" if prob >= max(0.80, threshold + 0.20) else "medium"
    elif (
        has_model_support
        and support_prob >= threshold
        and not safe_routine_message
        and (
            bool(dangerous_risk_keys & risk_keys)
            or 'urgency' in risk_keys
            or risk_score >= 1.4
        )
    ):
        main_label = "Phishing"
        level = "medium"
        display_prob = support_prob
        decision_basis = 'support_model_risk_override'
    else:
        main_label = "Legitimate"
        level = "review" if risk_score > 0 and near_boundary else "low"

    return {
        "label": main_label,
        "level": level,
        "review_recommended": review_recommended,
        "display_prob": display_prob,
        "model_prob": prob,
        "decision_basis": decision_basis,
        "support_prob": support_prob,
    }


def build_model_input(body: str, subject: str = '', sender_email: str = '', sender_context: str = '') -> str:
    parts = []
    sender = normalize_message_text(sender_email)
    sender_display = normalize_message_text(sender_context)
    subj = normalize_message_text(subject)
    body_text = normalize_message_text(body)

    # Unified model input
    if sender_display and sender_display != sender:
        parts.append(f'Sender: {sender_display}')
    if sender:
        parts.append(f'From: {sender}')
    if subj:
        parts.append(f'Subject: {subj}')
    if body_text:
        parts.append(body_text)

    return '\n\n'.join(parts)


def render_signal_group(title: str, items, icon: str = ''):
    if not items:
        return
    st.markdown(f'#### {title}')
    for item in items:
        prefix = f'{icon} ' if icon else ''
        st.markdown(f"{prefix}**{item['title']}** - {item['description']}")
        if item.get('matches'):
            examples = [truncate_display_value(match) for match in item['matches']]
            st.caption('Examples: ' + '; '.join(examples))


def useful_context_items(items):
    hidden_keys = {
        'sender_email_present',
        'email_address_present',
    }
    return [item for item in (items or []) if item.get('key') not in hidden_keys]


def truncate_display_value(value: str, limit: int = 90) -> str:
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(text) <= limit:
        return text
    return f'{text[:limit - 3]}...'


def get_risk_level(label: str, probability: float, threshold: float, review_recommended: bool = False) -> str:
    prob = float(probability)
    boundary_gap = abs(prob - float(threshold))

    if label == 'Phishing':
        if prob >= 0.80 or boundary_gap >= 0.18:
            return 'High'
        return 'Medium'

    if review_recommended:
        return 'Medium'
    if prob <= 0.25:
        return 'Very low'
    return 'Low'


def result_advice(label: str, risk_level: str, probability: float, threshold: float) -> str:
    if risk_level == 'Medium' and abs(float(probability) - float(threshold)) < 0.08:
        return 'Borderline result. Review the message manually before clicking links or sharing details.'
    if label == 'Phishing':
        if risk_level == 'High':
            return 'Do not click links or provide credentials. Verify through the official website.'
        return 'High risk. Treat this message as suspicious.'
    return 'Low risk, but review links before clicking.'


def verdict_confidence(label: str, probability: float) -> float:
    prob = float(probability)
    return 1.0 - prob if label == 'Legitimate' else prob


def render_result_summary(verdict: dict, risk_level: str, model_name: str, advice: str, probability: float):
    label = verdict['label']
    confidence = verdict_confidence(label, probability)
    summary = (
        f"**Verdict:** {label}\n\n"
        f"**Risk level:** {risk_level}\n\n"
        f"**Confidence in verdict:** {fmt_pct(confidence)}\n\n"
        f"**Model used:** {friendly_model_name(model_name)}\n\n"
        f"**Advice:** {advice}"
    )

    if label == 'Phishing' or risk_level == 'High':
        st.error(summary)
    elif risk_level == 'Medium':
        st.warning(summary)
    else:
        st.success(summary)

    st.progress(
        min(max(confidence, 0.0), 1.0),
        text=f'Confidence in verdict: {fmt_pct(confidence)}',
    )


def render_indicator_table(rules: dict):
    rows = [
        ('Urgency language', rules.get('has_urgency')),
        ('Credential request', rules.get('has_credential_request')),
        ('Threat language', rules.get('has_threat')),
        ('Security notice context', rules.get('has_security_notification')),
        ('URLs detected', rules.get('url_count', 0)),
        ('Email addresses detected', len(rules.get('email_addresses') or [])),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=['Indicator', 'Value']),
        use_container_width=True,
        hide_index=True,
    )

    urls = rules.get('urls') or []
    if urls:
        st.markdown('#### Detected URLs')
        st.dataframe(
            pd.DataFrame({'URL preview': [truncate_display_value(url, 100) for url in urls]}),
            use_container_width=True,
            hide_index=True,
        )
        st.code('\n'.join(urls), language='text')


def render_explanation_sections(
    *,
    summary: str,
    rules: dict,
    risk_items: list[dict],
    reassurance_items: list[dict],
    context_items: list[dict],
    expanded: bool = False,
):
    with st.expander('Why this result was given', expanded=expanded):
        safe_summary = escape(summary).replace('\n', '<br>')
        st.markdown(f"<div class='explanation-card'>{safe_summary}</div>", unsafe_allow_html=True)
        main_reasons = risk_items or reassurance_items or context_items
        if main_reasons:
            st.markdown('**Main reasons**')
            for item in main_reasons[:5]:
                st.markdown(f"- **{item['title']}**: {item['description']}")
        else:
            st.info('No strong rule-based indicators were found in the visible text. The result is mostly driven by the classifier output.')

    with st.expander('Detected links and indicators'):
        render_indicator_table(rules)
        render_signal_group('Warning signs', risk_items, '!')
        render_signal_group('Context that lowers risk', reassurance_items, '+')
        render_signal_group('Additional context', context_items, '')


def inject_custom_css():
    st.markdown(
        """
        <style>
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
            font-size: 2.25rem;
            font-weight: 700;
            line-height: 1.15;
            margin: 1.45rem 0 0.75rem 0;
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
            padding-top: 3.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_heading(title: str):
    st.markdown(f"<h1 class='page-heading'>{escape(title)}</h1>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_baseline_predictor():
    # Load once per session
    return BaselinePredictor()


@st.cache_resource(show_spinner=False)
def get_distilbert_predictor():
    # Load once per session
    if not distilbert_assets_available():
        raise RuntimeError('DistilBERT weights are not available in this deployment.')
    from src.inference.distilbert import DistilBertPredictor
    return DistilBertPredictor()


st.set_page_config(page_title='Veridexia', layout='wide')
inject_custom_css()
st.title('Veridexia: ML-Based Phishing Detection')

# Available runtime features
model_choices = available_model_choices()
distilbert_enabled = 'distilbert' in model_choices
gmail_enabled = GOOGLE_CREDENTIALS.exists()

# Session state defaults
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
    # Global controls
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
            help="Choose which classifier makes the prediction. DistilBERT is the project's main transformer-based model, while the baseline remains available for comparison.",
        )
    else:
        st.session_state.model_choice = 'distilbert' if distilbert_enabled else 'baseline'

    if not distilbert_enabled:
        st.caption('This deployment is using the baseline model for live predictions because the DistilBERT weights are not bundled here.')

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


# Main app tabs
tab_scan, tab_activity, tab_about, tab_guide = st.tabs(
    ['Scan Email', 'Recent Activity', 'About', 'Guide']
)

with tab_scan:
    # Main scan workflow
    render_page_heading('Scan Email')
    st.caption('Paste the body of an email or upload exported emails or text documents. The classifier uses any extracted text together with the sender email and subject when available.')

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
        email_text = st.text_area('Email Content', height=280, key='scan_email_text', label_visibility='collapsed')
        st.markdown('**Upload exported emails or text documents**')
        uploaded_files = st.file_uploader(
            'Upload exported emails or text documents',
            type=[
                'txt', 'text', 'md', 'rst', 'log', 'csv', 'tsv', 'json',
                'yaml', 'yml', 'ini', 'cfg', 'xml', 'html', 'htm',
                'docx', 'pdf', 'eml',
            ],
            accept_multiple_files=True,
            label_visibility='collapsed',
            help='Use exported .eml files or previously saved text-based documents. Screenshots and images are not supported.',
        )
        upload_context = build_upload_context(uploaded_files)
        st.caption('Safety note: avoid downloading or opening unknown live attachments.')

        if upload_context.items:
            st.caption(f'Loaded {len(upload_context.items)} uploaded file(s) for analysis.')
            for item in upload_context.items:
                extraction_mode = item.kind.upper()
                if item.text:
                    st.caption(f'{item.filename}: {extraction_mode} text extracted')
                else:
                    st.caption(f'{item.filename}: uploaded, but no readable text extracted yet')

        if upload_context.subject and not subject_line.strip():
            st.caption(f"Detected subject from upload: {upload_context.subject}")
        if upload_context.sender_email and not sender_email.strip():
            st.caption(f"Detected sender from upload: {upload_context.sender_email}")

        for warning in upload_context.warnings:
            st.info(warning)

        run = st.button('Analyze', type='primary', use_container_width=True)

    with right:
        # Output panel
        st.subheader('Results')

        if run:
            model_choice = st.session_state.model_choice
            threshold = float(st.session_state.threshold)
            effective_sender = (sender_email or '').strip() or upload_context.sender_email
            effective_sender_context = (
                st.session_state.scan_sender_context.strip()
                if isinstance(st.session_state.scan_sender_context, str)
                else ''
            )
            if effective_sender and effective_sender_context and effective_sender not in effective_sender_context:
                effective_sender_context = f'{effective_sender_context} <{effective_sender}>'
            effective_subject = (subject_line or '').strip() or upload_context.subject
            combined_body = '\n\n'.join(
                part for part in [email_text.strip(), upload_context.body_text] if part
            ).strip()
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
                st.warning('Paste some email content or upload a file with readable text first.')
                st.stop()

            with st.spinner(f'Analysing email with {friendly_model_name(model_choice)}...'):
                start = time.time()

                if model_choice == 'baseline':
                    # Baseline path
                    predictor = get_baseline_predictor()
                    result = predictor.predict(model_input, threshold=threshold)
                    evidence = baseline_evidence(model_input, predictor)
                    support_result = None
                else:
                    # DistilBERT + baseline support
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
            st.markdown("<p class='small-note'>The result card, explanation, and user-friendly signals will appear here after analysis.</p>", unsafe_allow_html=True)
            st.info('Example output: Likely phishing / Suspicious - review recommended / Likely legitimate')

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
                            render_signal_group('Context that lowers risk', reassurance_signals, '+')
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
        You can paste email text directly, import from Gmail, or upload readable files such as
        `.eml`, `.txt`, `.docx`, or `.pdf`.
        """
    )

    st.divider()
    st.markdown('### Understanding results')
    st.markdown(
        """
        Always read the explanation as well as the final phishing or legitimate label.
        Some legitimate emails, such as newsletters, job alerts, and account notices, may still trigger warning signs.
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
    st.warning('Do not download or open unknown live attachments just to analyse them in the demo.')


