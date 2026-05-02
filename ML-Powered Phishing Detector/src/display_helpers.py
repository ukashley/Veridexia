from __future__ import annotations

import re
from html import escape

import pandas as pd
import streamlit as st


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


def render_metrics_table(rows: list[dict]):
    # Used by the analysis/reporting view to present model metrics consistently.
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


def format_gmail_message_label(item: dict) -> str:
    # Compact label for the Gmail preview dropdown.
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


def activity_time_label(scanned_at: float | None) -> str:
    if not isinstance(scanned_at, (int, float)):
        return 'Unknown time'
    import time
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(float(scanned_at)))


def truncate_display_value(value: str, limit: int = 90) -> str:
    text = re.sub(r'\s+', ' ', str(value or '')).strip()
    if len(text) <= limit:
        return text
    return f'{text[:limit - 3]}...'


def render_signal_group(title: str, items, icon: str = ''):
    # Show evidence in human-readable groups rather than dumping raw dictionaries.
    if not items:
        return
    st.markdown(f'#### {title}')
    for item in items:
        prefix = f'{icon} ' if icon else ''
        st.markdown(f"{prefix}**{item['title']}** - {item['description']}")
        if item.get('matches'):
            examples = [truncate_display_value(match) for match in item['matches']]
            st.caption('Examples: ' + '; '.join(examples))


def render_reason_item(item: dict):
    st.markdown(f"- **{item['title']}**: {item['description']}")
    if item.get('matches'):
        examples = [truncate_display_value(match) for match in item['matches']]
        st.caption('Examples: ' + '; '.join(examples))


def verdict_confidence(label: str, probability: float) -> float:
    prob = float(probability)
    return 1.0 - prob if label == 'Legitimate' else prob


def render_result_summary(verdict: dict, risk_level: str, model_name: str, advice: str, probability: float):
    # First result view: keep it simple for users, then put evidence below.
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
    # User-visible evidence summary, not a raw model/debug dump.
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


def render_explanation_sections(
    *,
    summary: str,
    rules: dict,
    risk_items: list[dict],
    reassurance_items: list[dict],
    context_items: list[dict],
    expanded: bool = False,
):
    # Explanations are always available but collapsed by default to reduce clutter.
    with st.expander('Why this result was given', expanded=expanded):
        safe_summary = escape(summary).replace('\n', '<br>')
        st.markdown(f"<div class='explanation-card'>{safe_summary}</div>", unsafe_allow_html=True)
        main_reasons = risk_items or reassurance_items or context_items
        if main_reasons:
            st.markdown('**Main reasons**')
            for item in main_reasons[:5]:
                render_reason_item(item)
        else:
            st.info('No strong rule-based indicators were found in the visible text. The result is mostly driven by the classifier output.')

    with st.expander('Detected links and indicators'):
        render_indicator_table(rules)
        render_signal_group('Warning signs', risk_items, '!')
        render_signal_group('Additional context', context_items, '')
