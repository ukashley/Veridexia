# pyright: reportMissingImports=false

import sys
from pathlib import Path
import json
import time

import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.baseline import BaselinePredictor
from src.inference.distilbert import DistilBertPredictor
from src.explain.baseline_evidence import baseline_evidence
from src.explain.nlg import generate_explanation
from src.explain.rule_evidence import rule_based_evidence

DATASET_STATS = ROOT / 'data' / 'processed' / 'dataset_stats.json'
BASELINE_DIR = ROOT / 'models' / 'baseline'
DISTILBERT_DIR = ROOT / 'models' / 'distilbert'
RESULTS_DIR = ROOT / 'results' / 'external'

BASELINE_METRICS = RESULTS_DIR / 'trec06_baseline_metrics.json'
DISTILBERT_METRICS = RESULTS_DIR / 'trec06_distilbert_metrics.json'
BASELINE_CM_IMG = RESULTS_DIR / 'trec06_baseline_confusion_matrix.png'
DISTILBERT_CM_IMG = RESULTS_DIR / 'trec06_distilbert_confusion_matrix.png'
BASELINE_ROC_IMG = BASELINE_DIR / 'roc_curve.png'
BASELINE_FEAT_IMG = BASELINE_DIR / 'feature_importance.png'


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
        return 'â€”'


def get_external_kpis(m: dict):
    if not m or 'report' not in m:
        return None

    rep = m['report']
    cls1 = rep.get('1', {})
    cls0 = rep.get('0', {})
    macro = rep.get('macro avg', {})
    weighted = rep.get('weighted avg', {})

    return {
        'accuracy': m.get('accuracy'),
        'f1_macro': macro.get('f1-score'),
        'f1_weighted': weighted.get('f1-score'),
        'recall_phish': cls1.get('recall'),
        'precision_phish': cls1.get('precision'),
        'f1_phish': cls1.get('f1-score'),
        'recall_legit': cls0.get('recall'),
    }


def severity_color(level: str):
    return {
        'high': 'ðŸ”´ High risk',
        'medium': 'ðŸŸ  Moderate risk',
        'review': 'ðŸŸ¡ Needs review',
        'low': 'ðŸŸ¢ Low risk',
    }.get(level, 'ðŸŸ¡ Needs review')


def confidence_label(prob: float, threshold: float):
    gap = abs(float(prob) - float(threshold))
    if gap >= 0.25:
        return 'High'
    if gap >= 0.12:
        return 'Moderate'
    return 'Mixed'


def compute_user_verdict(result, rules: dict):
    prob = float(result.prob_phishing)
    threshold = float(result.threshold)

    no_strong_cues = (
        not bool(rules.get("has_credential_request"))
        and not bool(rules.get("has_threat"))
        and not bool(rules.get("has_urgency"))
        and int(rules.get("url_count", 0)) == 0
    )

    near_boundary = abs(prob - threshold) < 0.12
    review_recommended = near_boundary or (prob >= threshold and no_strong_cues)

    main_label = "Phishing" if prob >= threshold else "Legitimate"

    if prob >= threshold:
        level = "high" if prob >= max(0.80, threshold + 0.20) else "medium"
    else:
        level = "low"

    return {
        "label": main_label,
        "level": level,
        "review_recommended": review_recommended,
    }


def build_model_input(body: str, subject: str = '', sender_email: str = '') -> str:
    parts = []
    sender = (sender_email or '').strip()
    subj = (subject or '').strip()
    body_text = (body or '').strip()

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
st.title('Veridexia - An AI-Based Phishing Detector & Explainer')

if 'model_choice' not in st.session_state:
    st.session_state.model_choice = 'distilbert'
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.65
if 'show_explain' not in st.session_state:
    st.session_state.show_explain = True
if 'advanced_mode' not in st.session_state:
    st.session_state.advanced_mode = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.header('Quick Controls')
    st.session_state.model_choice = st.selectbox(
        'Default Model',
        ['distilbert', 'baseline'],
        index=0 if st.session_state.model_choice == 'distilbert' else 1,
        help='DistilBERT is usually stronger in-domain; the baseline can generalise better externally.',
    )
    st.session_state.threshold = st.slider(
        'Phishing threshold',
        0.0, 1.0, float(st.session_state.threshold), 0.01,
        help='Lower = catches more phishing but raises false positives. Higher = stricter.',
    )
    st.session_state.show_explain = st.toggle('Show explanation', value=st.session_state.show_explain)
    st.session_state.advanced_mode = st.toggle('Advanced mode', value=st.session_state.advanced_mode)

    st.divider()
    st.caption('Files status')
    st.write('âœ… dataset_stats.json' if DATASET_STATS.exists() else 'âš ï¸ dataset_stats.json missing')
    st.write('âœ… baseline external metrics' if BASELINE_METRICS.exists() else 'âš ï¸ baseline external metrics missing')
    st.write('âœ… distilbert external metrics' if DISTILBERT_METRICS.exists() else 'âš ï¸ distilbert external metrics missing')


tab_dashboard, tab_scan, tab_analysis, tab_settings = st.tabs(
    ['Dashboard', 'Scan Email', 'Model Analysis', 'Settings']
)

with tab_dashboard:
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
            k1.metric('DistilBERT F1 (phish)', fmt_pct(d_kpi.get('f1_phish')))
            k2.metric('DistilBERT Recall (phish)', fmt_pct(d_kpi.get('recall_phish')))
        else:
            k1.metric('DistilBERT F1 (phish)', 'â€”')
            k2.metric('DistilBERT Recall (phish)', 'â€”')

        if b_kpi:
            k3.metric('Baseline F1 (phish)', fmt_pct(b_kpi.get('f1_phish')))
            k4.metric('Baseline Accuracy', fmt_pct(b_kpi.get('accuracy')))
        else:
            k3.metric('Baseline F1 (phish)', 'â€”')
            k4.metric('Baseline Accuracy', 'â€”')

        st.divider()
        st.subheader('Dataset Snapshot')
        if stats:
            ds = stats.get('dataset_info', {})
            cd = stats.get('class_distribution', {})
            fs = stats.get('feature_statistics', {})

            cA, cB, cC, cD = st.columns(4)
            cA.metric('Total emails', f"{ds.get('total_samples', 'â€”'):,}" if isinstance(ds.get('total_samples'), int) else 'â€”')
            cB.metric('Train / Val / Test', f"{ds.get('train_samples','â€”')} / {ds.get('val_samples','â€”')} / {ds.get('test_samples','â€”')}")
            cC.metric('Phishing ratio', fmt_pct(cd.get('phishing_ratio')))
            cD.metric('Imbalance ratio', f"{cd.get('imbalance_ratio', 'â€”'):.3f}" if isinstance(cd.get('imbalance_ratio'), (int, float)) else 'â€”')

            st.caption('Feature averages from preprocessing:')
            f1c, f2c, f3c = st.columns(3)
            f1c.metric('Avg URLs (phishing)', f"{fs.get('avg_urls_phishing', 'â€”'):.2f}" if isinstance(fs.get('avg_urls_phishing'), (int, float)) else 'â€”')
            f2c.metric('Avg URLs (legit)', f"{fs.get('avg_urls_legitimate', 'â€”'):.2f}" if isinstance(fs.get('avg_urls_legitimate'), (int, float)) else 'â€”')
            f3c.metric('Avg urgency (phishing)', f"{fs.get('avg_urgency_phishing', 'â€”'):.2f}" if isinstance(fs.get('avg_urgency_phishing'), (int, float)) else 'â€”')
        else:
            st.info('dataset_stats.json not found yet. Run your dataset preparation script to generate it.')

        if st.session_state.last_result:
            st.divider()
            st.subheader('Last Scan')
            st.write(st.session_state.last_result)

    with col2:
        st.subheader('Key Visuals')
        if BASELINE_CM_IMG.exists():
            st.image(str(BASELINE_CM_IMG), caption='Baseline confusion matrix (TREC-06)', use_container_width=True)
        else:
            st.caption('Baseline confusion matrix image not found.')

        if DISTILBERT_CM_IMG.exists():
            st.image(str(DISTILBERT_CM_IMG), caption='DistilBERT confusion matrix (TREC-06)', use_container_width=True)
        else:
            st.caption('DistilBERT confusion matrix image not found.')

with tab_scan:
    st.subheader('Scan Email')
    st.caption('Paste the body of an email and optionally the subject line. The classifier makes the decision; the rule layer explains visible warning signs.')

    left, right = st.columns([1.15, 1])

    with left:
        model_choice = st.selectbox(
            'Model',
            ['distilbert', 'baseline'],
            index=0 if st.session_state.model_choice == 'distilbert' else 1,
        )
        threshold = st.slider('Phishing threshold', 0.0, 1.0, float(st.session_state.threshold), 0.01)
        show_explain = st.checkbox('Show explanation', value=st.session_state.show_explain)
        advanced_mode = st.checkbox('Advanced mode', value=st.session_state.advanced_mode)
        sender_email = st.text_input('Sender email (optional)')
        subject_line = st.text_input('Subject (optional)')
        email_text = st.text_area('Paste email content here', height=280)

        run = st.button('Analyze', type='primary', use_container_width=True)

    with right:
        st.subheader('Results')

        if run:
            model_input = build_model_input(email_text, subject=subject_line, sender_email=sender_email)
            if not model_input.strip():
                st.warning('Paste some email content first.')
                st.stop()

            start = time.time()

            if model_choice == 'baseline':
                predictor = get_baseline_predictor()
                result = predictor.predict(model_input, threshold=threshold)
                evidence = baseline_evidence(model_input, predictor)
            else:
                predictor = get_distilbert_predictor()
                result = predictor.predict(model_input, threshold=threshold)
                evidence = None

            rules = rule_based_evidence(email_text, sender_email=sender_email, subject=subject_line)
            p_phish = float(result.prob_phishing)
            ms = (time.time() - start) * 1000.0
            summary = generate_explanation(result, evidence=evidence, rule_evidence=rules)

            verdict = compute_user_verdict(result, rules)
            review_text = ' | Manual review recommended' if verdict['review_recommended'] else ''

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">{verdict['label']}</div>
                    <div class="result-meta">
                        Probability: {p_phish:.2f} | Threshold: {result.threshold:.2f}
                        | Confidence: {confidence_label(p_phish, result.threshold)}{review_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if show_explain:
                st.markdown(f"<div class='explanation-card'>{summary}</div>", unsafe_allow_html=True)

                risk_items = rules.get('signals', [])
                reassurance_items = rules.get('reassurance_signals', [])
                context_items = rules.get('context_signals', [])

                if risk_items or reassurance_items or context_items:
                    st.markdown('### Why this result was given')
                    render_signal_group('Warning signs', risk_items, '!')
                    render_signal_group('Context that lowers risk', reassurance_items, '+')
                    render_signal_group('Additional context', context_items, '')
                else:
                    st.info('No strong rule-based indicators were found in the visible text. The result is mostly driven by the classifier output.')

            st.progress(min(max(p_phish, 0.0), 1.0))
            st.caption(f'Model: {result.model_name} | Inference time: {ms:.1f} ms')

            if advanced_mode:
                st.divider()
                st.markdown('### Advanced details')
                st.json({
                    'label': result.label,
                    'prob_phishing': p_phish,
                    'threshold': result.threshold,
                    'model_name': result.model_name,
                    'rule_evidence': rules,
                    'model_evidence': evidence,
                })

            if model_choice == 'distilbert':
                st.caption('Token-level attribution can be added later, but the current explanation layer is intentionally model-agnostic.')

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
    st.subheader('Model Analysis & Comparison (External: TREC-06)')

    baseline_metrics = load_json(BASELINE_METRICS)
    distilbert_metrics = load_json(DISTILBERT_METRICS)
    rows = []

    if baseline_metrics:
        k = get_external_kpis(baseline_metrics)
        if k:
            rows.append({
                'Model': 'Baseline (TF-IDF + Logistic Regression)',
                'Accuracy': k.get('accuracy'),
                'Precision (phish)': k.get('precision_phish'),
                'Recall (phish)': k.get('recall_phish'),
                'F1 (phish)': k.get('f1_phish'),
            })

    if distilbert_metrics:
        k = get_external_kpis(distilbert_metrics)
        if k:
            rows.append({
                'Model': 'DistilBERT',
                'Accuracy': k.get('accuracy'),
                'Precision (phish)': k.get('precision_phish'),
                'Recall (phish)': k.get('recall_phish'),
                'F1 (phish)': k.get('f1_phish'),
            })

    if rows:
        df = pd.DataFrame(rows)
        for col in ['Accuracy', 'Precision (phish)', 'Recall (phish)', 'F1 (phish)']:
            df[col] = df[col].apply(lambda x: fmt_pct(x) if isinstance(x, (int, float)) else 'â€”')
        html = df.to_html(index=False)
        st.markdown(f"<div class='big-table'>{html}</div>", unsafe_allow_html=True)
    else:
        st.info('External metrics not found yet. Add files to results/external/.')

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.subheader('Baseline visual (external)')
        if BASELINE_CM_IMG.exists():
            st.image(str(BASELINE_CM_IMG), use_container_width=True)
        else:
            st.caption('Baseline confusion matrix image not found.')

    with c2:
        st.subheader('DistilBERT visual (external)')
        if DISTILBERT_CM_IMG.exists():
            st.image(str(DISTILBERT_CM_IMG), use_container_width=True)
        else:
            st.caption('DistilBERT confusion matrix image not found.')

    st.caption('ROC curve and feature-importance plots are hidden in external-validation view.')

with tab_settings:
    st.subheader('Settings')
    st.write('These settings change default behaviour across the app and help make the interface easier to demo.')

    s1, s2 = st.columns([1, 1])

    with s1:
        st.session_state.model_choice = st.radio(
            'Default model',
            ['distilbert', 'baseline'],
            index=0 if st.session_state.model_choice == 'distilbert' else 1,
        )
        st.session_state.threshold = st.slider('Default threshold', 0.0, 1.0, float(st.session_state.threshold), 0.01)
        st.session_state.show_explain = st.checkbox('Show explanations by default', value=st.session_state.show_explain)
        st.session_state.advanced_mode = st.checkbox('Enable advanced mode by default', value=st.session_state.advanced_mode)

    with s2:
        st.markdown('### Recommended presets')
        if st.button('ðŸ”’ Safer (fewer false positives)'):
            st.session_state.threshold = 0.70
        if st.button('ðŸŽ£ More sensitive (catch more phishing)'):
            st.session_state.threshold = 0.40
        if st.button('ðŸ§ª Debug-friendly'):
            st.session_state.advanced_mode = True
            st.session_state.show_explain = True

        st.divider()
        st.markdown('### File locations')
        st.code(
            '\n'.join([
                f'dataset_stats: {DATASET_STATS}',
                f'baseline metrics: {BASELINE_METRICS}',
                f'distilbert metrics: {DISTILBERT_METRICS}',
                f'baseline cm: {BASELINE_CM_IMG}',
                f'distilbert cm: {DISTILBERT_CM_IMG}',
            ]),
            language='text',
        )


