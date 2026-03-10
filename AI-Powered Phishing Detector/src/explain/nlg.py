from __future__ import annotations


def _join_labels(items):
    items = [item for item in items if item]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f'{items[0]} and {items[1]}'
    return ', '.join(items[:-1]) + f', and {items[-1]}'


def _confidence_band(prob_phishing: float, threshold: float) -> str:
    gap = abs(float(prob_phishing) - float(threshold))
    if gap >= 0.25:
        return 'high'
    if gap >= 0.12:
        return 'moderate'
    return 'mixed'


def generate_explanation(result, evidence=None, rule_evidence=None):
    prob = float(result.prob_phishing)
    threshold = float(result.threshold)
    confidence = _confidence_band(prob, threshold)

    rules = rule_evidence or {}
    risk_titles = [s.get('title', '').lower() for s in rules.get('signals', [])][:3]
    reassurance_titles = [s.get('title', '').lower() for s in rules.get('reassurance_signals', [])][:2]
    context_titles = [s.get('title', '').lower() for s in rules.get('context_signals', [])][:2]
    has_credential_request = bool(rules.get('has_credential_request'))
    risk_score = float(rules.get('risk_score', 0.0))

    parts = []

    if result.label == 1:
        if risk_titles:
            parts.append(
                f'This email was flagged as phishing because it shows warning signs such as {_join_labels(risk_titles)}.'
            )
        else:
            parts.append(
                'This email was flagged as phishing because the model detected language patterns that are associated with phishing, even though only limited explicit rule-based cues were found.'
            )

        if reassurance_titles:
            parts.append(
                f'Some wording also suggested a more benign interpretation, including {_join_labels(reassurance_titles)}, so this should be treated as a cautious warning rather than an absolute conclusion.'
            )

        if not risk_titles and context_titles:
            parts.append(
                f'The visible text mainly provided context such as {_join_labels(context_titles)} rather than strong phishing indicators.'
            )

        parts.append(
            'Avoid clicking links or sharing passwords, one-time codes, or account details until the message has been verified through an official website or trusted contact channel.'
        )

    else:
        if has_credential_request or risk_score >= 1.6:
            parts.append(
                f'This email was classified as legitimate overall, but it still contains caution signals such as {_join_labels(risk_titles)}.'
            )
            parts.append(
                'Because the wording still includes suspicious elements, it should be checked manually before any action is taken.'
            )
        elif reassurance_titles:
            parts.append(
                f'This email was classified as legitimate because it reads more like a normal service or security message, with context such as {_join_labels(reassurance_titles)}.'
            )
        else:
            parts.append(
                'This email was classified as legitimate because it does not strongly match common phishing patterns such as credential harvesting, urgency pressure, or threatening account language.'
            )

        if context_titles and not reassurance_titles:
            parts.append(
                f'The message also contains neutral context such as {_join_labels(context_titles)}, which does not by itself indicate phishing.'
            )

    if confidence == 'high':
        parts.append('Model confidence for this decision was high.')
    elif confidence == 'moderate':
        parts.append('Model confidence for this decision was moderate.')
    else:
        parts.append('The indicators for this message were mixed, so manual caution is still advisable.')

    if evidence and isinstance(evidence, dict) and evidence.get('top_terms'):
        terms = [term for term, _ in evidence['top_terms'][:5]]
        if terms:
            parts.append(f'For the baseline model, influential terms included: {", ".join(terms)}.')

    return ' '.join(part for part in parts if part).strip()
