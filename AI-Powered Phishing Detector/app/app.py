# pyright: reportMissingImports=false

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
from src.inference.distilbert import DistilBertPredictor
from src.inference.upload_extractors import build_upload_context
from src.explain.baseline_evidence import baseline_evidence
from src.explain.nlg import generate_explanation
from src.explain.rule_evidence import rule_based_evidence

DATASET_STATS = ROOT / 'data' / 'processed' / 'dataset_stats.json'
BASELINE_DIR = ROOT / 'models' / 'baseline'
DISTILBERT_DIR = ROOT / 'models' / 'distilbert'
RESULTS_DIR = ROOT / 'results' / 'external'

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
    'Low': 0.77,
    'Balanced': 0.65,
    'High': 0.55,
}
INVISIBLE_TRANSLATION = str.maketrans('', '', '\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff')
def load_json(path: Path):
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

    near_boundary = abs(prob - threshold) < 0.12
    review_recommended = near_boundary or (prob >= threshold and no_strong_cues)
    benign_context = (
        'sender_email_present' in context_keys
        and 'email_address_present' in context_keys
        and 'domain_mismatch' not in risk_keys
        and 'suspicious_link' not in risk_keys
    )
    clean_message = no_strong_cues and risk_score <= 0 and 'domain_mismatch' not in risk_keys and 'suspicious_link' not in risk_keys
    low_severity_risk_keys = {'payment_language', 'generic_greeting'}
    low_severity_only = bool(risk_keys) and risk_keys.issubset(low_severity_risk_keys)
    routine_message_keys = {
        'transactional_notification',
        'account_administration_notice',
        'newsletter_context',
        'job_alert_context',
    }
    transactional_reassurance = bool(
        (routine_message_keys | {'sender_brand_match', 'domain_match'}) & reassurance_keys
    )
    strong_benign_reassurance = bool(
        routine_message_keys & reassurance_keys
    ) and bool({'sender_brand_match', 'domain_match'} & reassurance_keys)
    display_prob = prob
    decision_basis = 'model'

    if prob >= threshold and near_boundary and no_strong_cues and risk_score <= 0 and (
        benign_context or 'domain_match' in reassurance_keys
    ):
        main_label = "Legitimate"
        level = "review"
        display_prob = max(0.0, threshold - 0.01)
        decision_basis = 'hybrid_override'
    elif (
        prob >= threshold
        and support_prob is not None
        and support_prob < threshold
        and no_strong_cues
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
        and clean_message
        and ('sender_email_present' in context_keys or benign_context or 'domain_match' in reassurance_keys)
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
        and no_strong_cues
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
        and no_strong_cues
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
            st.caption('Examples: ' + '; '.join(item['matches']))


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
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_baseline_predictor():
    return BaselinePredictor()


@st.cache_resource
def get_distilbert_predictor():
    return DistilBertPredictor()


st.set_page_config(page_title='Veridexia', layout='wide')
inject_custom_css()
st.title('Veridexia: ML-Based Phishing Detection')

if 'model_choice' not in st.session_state:
    st.session_state.model_choice = 'distilbert'
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.65
if 'detection_sensitivity' not in st.session_state:
    st.session_state.detection_sensitivity = sensitivity_for_threshold(st.session_state.threshold)
if 'show_explain' not in st.session_state:
    st.session_state.show_explain = True
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

with st.sidebar:
    st.header('Quick Controls')
    st.session_state.model_choice = st.selectbox(
        'Model',
        ['baseline', 'distilbert'],
        index=0 if st.session_state.model_choice == 'baseline' else 1,
        help='Choose which classifier makes the prediction. Baseline is the default because it currently performs better on the external test set. DistilBERT remains available for comparison.',
    )
    st.session_state.show_explain = st.toggle(
        'Show explanation',
        value=st.session_state.show_explain,
        help='Show the explanation and supporting evidence under the result.',
    )
    st.session_state.advanced_mode = st.toggle(
        'Advanced mode',
        value=st.session_state.advanced_mode,
        help='Show advanced controls such as detection sensitivity and technical details.',
    )
    if st.session_state.advanced_mode:
        st.session_state.detection_sensitivity = st.select_slider(
            'Detection sensitivity',
            options=list(SENSITIVITY_THRESHOLDS.keys()),
            value=st.session_state.detection_sensitivity,
            help='Low sensitivity is stricter and reduces false alarms. High sensitivity catches more suspicious emails but may flag more legitimate messages.',
        )
        st.caption(
            f"Current phishing threshold: {threshold_for_sensitivity(st.session_state.detection_sensitivity):.2f}"
        )

    st.session_state.threshold = threshold_for_sensitivity(st.session_state.detection_sensitivity)


tab_scan, tab_overview, tab_analysis, tab_settings = st.tabs(
    ['Scan Email', 'Dashboard', 'Model Analysis', 'Settings']
)

with tab_overview:
    col1, col2 = st.columns([1.2, 1])

    baseline_metrics = load_json(BASELINE_METRICS)
    distilbert_metrics = load_json(DISTILBERT_METRICS)
    stats = load_json(DATASET_STATS)

    with col1:
        st.subheader('System Overview')

        k1, k2, k3, k4 = st.columns(4)
        b_kpi = get_external_kpis(baseline_metrics)
        d_kpi = get_external_kpis(distilbert_metrics)

        if d_kpi:
            k1.metric('DistilBERT Accuracy', fmt_pct(d_kpi.get('accuracy')))
            k2.metric('DistilBERT Recall (phish)', fmt_pct(d_kpi.get('recall_phish')))
        else:
            k1.metric('DistilBERT Accuracy', '-')
            k2.metric('DistilBERT Recall (phish)', '-')

        if b_kpi:
            k3.metric('Baseline Accuracy', fmt_pct(b_kpi.get('accuracy')))
            k4.metric('Baseline Recall (phish)', fmt_pct(b_kpi.get('recall_phish')))
        else:
            k3.metric('Baseline Accuracy', '-')
            k4.metric('Baseline Recall (phish)', '-')

        st.divider()
        st.subheader('Dataset Snapshot')
        if stats:
            ds = stats.get('dataset_info', {})
            cd = stats.get('class_distribution', {})
            fs = stats.get('feature_statistics', {})

            cA, cB, cC, cD = st.columns(4)
            cA.metric('Total emails', fmt_count(ds.get('total_samples')))
            cB.metric('Train', fmt_count(ds.get('train_samples')))
            cC.metric('Val', fmt_count(ds.get('val_samples')))
            cD.metric('Test', fmt_count(ds.get('test_samples')))

            st.metric('Phishing ratio', fmt_pct(cd.get('phishing_ratio')))

            f1c, f2c, f3c = st.columns(3)
            f1c.metric('Avg URLs (phishing)', f"{fs.get('avg_urls_phishing', '-'):.2f}" if isinstance(fs.get('avg_urls_phishing'), (int, float)) else '-')
            f2c.metric('Avg URLs (legit)', f"{fs.get('avg_urls_legitimate', '-'):.2f}" if isinstance(fs.get('avg_urls_legitimate'), (int, float)) else '-')
            f3c.metric('Avg urgency (phishing)', f"{fs.get('avg_urgency_phishing', '-'):.2f}" if isinstance(fs.get('avg_urgency_phishing'), (int, float)) else '-')
        else:
            st.info('dataset_stats.json not found yet. Run your dataset preparation script to generate it.')

    with col2:
        st.subheader('Last Scan')
        if st.session_state.last_result:
            last = st.session_state.last_result
            verdict_label = last.get('display_verdict', 'Unknown')
            model_label = friendly_model_name(last.get('model', ''))
            prob_value = last.get('phishing_probability')
            threshold_value = float(last.get('threshold', st.session_state.threshold))
            confidence_value = (
                confidence_label(prob_value, threshold_value)
                if isinstance(prob_value, (int, float))
                else 'Unknown'
            )
            inference_ms = last.get('inference_ms')

            k1, k2 = st.columns(2)
            k1.metric('Result', verdict_label)
            k2.metric('Model', model_label)

            k3, k4 = st.columns(2)
            k3.metric('Confidence', confidence_value)
            k4.metric(
                'Speed',
                f"{float(inference_ms):.0f} ms" if isinstance(inference_ms, (int, float)) else '-'
            )

            if isinstance(prob_value, (int, float)):
                st.progress(min(max(float(prob_value), 0.0), 1.0))
                st.caption(f'Estimated phishing probability: {fmt_pct(prob_value)}')

            if verdict_label == 'Phishing':
                st.warning('The last scanned email looked suspicious and should be verified before any action is taken.')
            elif confidence_value == 'Mixed':
                st.info('The last scanned email was treated as legitimate, but it was close to the review boundary.')
            else:
                st.success('The last scanned email looked legitimate based on the current checks.')
        else:
            st.info('No emails have been scanned yet. Use the Scan Email tab to analyze a message.')

with tab_scan:
    st.subheader('Scan Email')
    st.caption('Paste the body of an email or upload exported email files or previously saved readable documents. Any extracted text is analysed together with the sender email and subject when available.')

    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('**Sender email**')
        sender_email = st.text_input('Sender email', key='scan_sender_email', label_visibility='collapsed')
        st.markdown('**Subject**')
        subject_line = st.text_input('Subject', key='scan_subject', label_visibility='collapsed')
        st.markdown('**Email Content**')
        email_text = st.text_area('Email Content', height=280, key='scan_email_text', label_visibility='collapsed')
        st.markdown('**Upload exported emails or readable documents**')
        uploaded_files = st.file_uploader(
            'Upload exported emails or readable documents',
            type=[
                'txt', 'text', 'md', 'rst', 'log', 'csv', 'tsv', 'json',
                'yaml', 'yml', 'ini', 'cfg', 'xml', 'html', 'htm',
                'docx', 'pdf', 'eml',
                'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp', 'tif', 'tiff',
            ],
            accept_multiple_files=True,
            label_visibility='collapsed',
            help='Do not open or run suspicious attachments just to analyse them here. Where possible, use exported .eml files or previously saved readable documents.',
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
        st.subheader('Results')

        if run:
            model_choice = st.session_state.model_choice
            threshold = float(st.session_state.threshold)
            show_explain = bool(st.session_state.show_explain)
            advanced_mode = bool(st.session_state.advanced_mode)
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

            start = time.time()

            if model_choice == 'baseline':
                predictor = get_baseline_predictor()
                result = predictor.predict(model_input, threshold=threshold)
                evidence = baseline_evidence(model_input, predictor)
                support_result = None
                support_evidence = None
            else:
                predictor = get_distilbert_predictor()
                result = predictor.predict(model_input, threshold=threshold)
                evidence = None
                support_predictor = get_baseline_predictor()
                support_result = support_predictor.predict(model_input, threshold=threshold)
                support_evidence = baseline_evidence(model_input, support_predictor)

            rules = rule_based_evidence(combined_body, sender_email=effective_sender, subject=effective_subject)
            verdict = compute_user_verdict(result, rules, support_result=support_result)
            p_phish = float(verdict.get('display_prob', result.prob_phishing))
            ms = (time.time() - start) * 1000.0
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

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">{verdict['label']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if verdict['review_recommended']:
                st.warning('Manual review is recommended for this message.')

            if show_explain:
                safe_summary = escape(summary).replace('\n', '<br>')
                st.markdown(f"<div class='explanation-card'>{safe_summary}</div>", unsafe_allow_html=True)

                risk_items = rules.get('signals', [])
                reassurance_items = rules.get('reassurance_signals', [])
                context_items = useful_context_items(rules.get('context_signals', []))

                if risk_items or reassurance_items or context_items:
                    st.markdown('### Why this result was given')
                    render_signal_group('Warning signs', risk_items, '!')
                    render_signal_group('Context that lowers risk', reassurance_items, '+')
                    render_signal_group('Additional context', context_items, '')
                else:
                    st.info('No strong rule-based indicators were found in the visible text. The result is mostly driven by the classifier output.')

            st.progress(min(max(p_phish, 0.0), 1.0))
            st.caption(f'Model: {result.model_name} | Inference time: {ms:.1f} ms')
            if uploaded_files:
                st.caption(f'Uploaded sources included: {len(uploaded_files)}')

            if advanced_mode:
                st.divider()
                st.markdown('### Advanced details')
                st.json({
                    'model_label': result.label,
                    'final_label': verdict['label'],
                    'prob_phishing': float(result.prob_phishing),
                    'decision_probability': p_phish,
                    'support_probability': verdict.get('support_prob'),
                    'threshold': result.threshold,
                    'model_name': result.model_name,
                    'decision_basis': verdict.get('decision_basis'),
                    'effective_sender_email': effective_sender,
                    'effective_subject': effective_subject,
                    'uploaded_files': [item.filename for item in upload_context.items],
                    'rule_evidence': rules,
                    'model_evidence': evidence,
                    'support_model_evidence': support_evidence,
                })

            if model_choice == 'distilbert':
                st.caption('Token-level attribution can be added later. The explanation currently uses model output, rule signals, and any available support evidence.')

            st.session_state.last_result = {
                'model': result.model_name,
                'threshold': float(threshold),
                'display_verdict': verdict['label'],
                'phishing_probability': float(p_phish),
                'risk_score': float(rules.get('risk_score', 0.0)),
                'inference_ms': float(ms),
            }
        else:
            st.markdown("<p class='small-note'>The result card, explanation, and user-friendly signals will appear here after analysis.</p>", unsafe_allow_html=True)
            st.info('Example output: Likely phishing / Suspicious - review recommended / Likely legitimate')

with tab_analysis:
    st.markdown('<h2>Model Analysis & Comparison</h2>', unsafe_allow_html=True)

    if MODEL_COMPARISON_IMG.exists():
        st.markdown('### Main comparison figure')
        st.image(str(MODEL_COMPARISON_IMG), use_container_width=True)
        st.divider()

    baseline_internal_metrics = load_json(BASELINE_INTERNAL_METRICS)
    distilbert_internal_metrics = load_json(DISTILBERT_INTERNAL_METRICS)
    baseline_metrics = load_json(BASELINE_METRICS)
    distilbert_metrics = load_json(DISTILBERT_METRICS)
    training_rows = []
    external_rows = []

    if baseline_internal_metrics:
        k = get_internal_kpis(baseline_internal_metrics)
        if k:
            training_rows.append({
                'Model': 'Baseline (TF-IDF + Logistic Regression)',
                'Accuracy': k.get('accuracy'),
                'Precision (phish)': k.get('precision_phish'),
                'Recall (phish)': k.get('recall_phish'),
                'F1 (phish)': k.get('f1_phish'),
                'False positive rate': k.get('false_positive_rate'),
                'False negative rate': k.get('false_negative_rate'),
            })

    if distilbert_internal_metrics:
        k = get_internal_kpis(distilbert_internal_metrics)
        if k:
            training_rows.append({
                'Model': 'DistilBERT',
                'Accuracy': k.get('accuracy'),
                'Precision (phish)': k.get('precision_phish'),
                'Recall (phish)': k.get('recall_phish'),
                'F1 (phish)': k.get('f1_phish'),
                'False positive rate': k.get('false_positive_rate'),
                'False negative rate': k.get('false_negative_rate'),
            })

    if baseline_metrics:
        k = get_external_kpis(baseline_metrics)
        if k:
            external_rows.append({
                'Model': 'Baseline (TF-IDF + Logistic Regression)',
                'Accuracy': k.get('accuracy'),
                'Precision (phish)': k.get('precision_phish'),
                'Recall (phish)': k.get('recall_phish'),
                'F1 (phish)': k.get('f1_phish'),
                'False positive rate': k.get('false_positive_rate'),
                'False negative rate': k.get('false_negative_rate'),
            })

    if distilbert_metrics:
        k = get_external_kpis(distilbert_metrics)
        if k:
            external_rows.append({
                'Model': 'DistilBERT',
                'Accuracy': k.get('accuracy'),
                'Precision (phish)': k.get('precision_phish'),
                'Recall (phish)': k.get('recall_phish'),
                'F1 (phish)': k.get('f1_phish'),
            'False positive rate': k.get('false_positive_rate'),
            'False negative rate': k.get('false_negative_rate'),
        })

    if training_rows:
        st.markdown('### Model training results (Kaggle)')
        render_metrics_table(training_rows)
        st.caption('These are the original split results saved from the model training artefacts. They provide an in-domain benchmark before external validation on a separate dataset.')
        st.divider()

    st.markdown('### External validation results (TREC-06)')
    if external_rows:
        render_metrics_table(external_rows)
        st.caption('These results show model performance on a separate external dataset and therefore provide a more realistic indication of generalisation')
    else:
        st.info('External metrics not found yet. Add files to results/external/.')

    st.divider()
    st.markdown('### Confusion matrices: model training results (Kaggle)')
    c1, c2 = st.columns(2)

    with c1:
        if BASELINE_INTERNAL_CM_IMG.exists():
            st.image(str(BASELINE_INTERNAL_CM_IMG), use_container_width=True)
        else:
            st.caption('Baseline internal confusion matrix image not found.')

    with c2:
        if DISTILBERT_INTERNAL_CM_IMG.exists():
            st.image(str(DISTILBERT_INTERNAL_CM_IMG), use_container_width=True)
        else:
            st.caption('DistilBERT internal confusion matrix image not found.')

    st.divider()
    st.markdown('### Confusion matrices: external validation (TREC-06)')
    c1, c2 = st.columns(2)

    with c1:
        if BASELINE_CM_IMG.exists():
            st.image(str(BASELINE_CM_IMG), use_container_width=True)
        else:
            st.caption('Baseline confusion matrix image not found.')

    with c2:
        if DISTILBERT_CM_IMG.exists():
            st.image(str(DISTILBERT_CM_IMG), use_container_width=True)
        else:
            st.caption('DistilBERT confusion matrix image not found.')

with tab_settings:
    st.subheader('Settings')
    st.write('This page summarises the current app profile and gives a few practical tips for cleaner scans.')

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Active model', friendly_model_name(st.session_state.model_choice))
    c2.metric('Sensitivity', st.session_state.detection_sensitivity)
    c3.metric('Explanation', 'On' if st.session_state.show_explain else 'Off')
    c4.metric('Advanced', 'On' if st.session_state.advanced_mode else 'Off')
    if st.session_state.advanced_mode:
        st.caption(f"Underlying phishing threshold: {float(st.session_state.threshold):.2f}")

    st.divider()
    st.markdown('### Scan tips')
    st.markdown(
        """
        - Include the sender email and subject when you have them. They help the scan use more context.
        - Use the full message where possible rather than a short extract.
        - Uploaded files work best for exported emails or readable documents such as `.eml`, `.txt`, `.docx`, or `.pdf`.
        - Avoid downloading or opening unknown live attachments just to analyse them in the demo.
        - Newsletters, job alerts, and account notices can still trigger review warnings, so read the explanation as well as the final label.
        - Treat `review recommended` as a caution signal, even when the final label is legitimate.
        """
    )


