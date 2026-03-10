from __future__ import annotations

import re
from urllib.parse import urlparse

URL_RE = re.compile(r'https?://\S+|www\.\S+', re.I)
EMAIL_RE = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)

CREDENTIAL_REQUEST_PATTERNS = [
    r'\b(enter|confirm|verify|provide|submit|update|share|send)\b.{0,35}\b(password|passcode|otp|one[- ]time code|verification code|login details?|credentials?)\b',
    r'\b(password|otp|credentials?|login details?)\b.{0,20}\b(required|needed|must be confirmed|must be verified|must be updated)\b',
    r'\bclick\b.{0,25}\b(login|sign in|verify|confirm|validate)\b',
]

SECURITY_NOTIFICATION_PATTERNS = [
    r'\byour password (was|has been) (changed|reset|updated)\b',
    r'\bif you did not (make|request|initiate|authorise|authorize) this (change|reset|request)\b',
    r'\bcontact (support|customer service|the helpdesk|the service desk)\b',
    r'\bwe will never ask for your password\b',
]

URGENCY_PATTERNS = [
    r'\burgent\b',
    r'\bimmediately\b',
    r'\basap\b',
    r'\baction required\b',
    r'\bact now\b',
    r'\bwithin \d+ (minutes?|hours?|days?)\b',
    r'\bexpires? (today|soon|within)\b',
]

THREAT_PATTERNS = [
    r'\baccount (will be )?(suspended|locked|disabled|terminated)\b',
    r'\bfailure to comply\b',
    r'\bpermanent suspension\b',
    r'\bunauthori[sz]ed activity\b',
    r'\bsecurity breach\b',
]

PAYMENT_PATTERNS = [
    r'\b(payment|invoice|refund|wire transfer|gift card|bank transfer)\b',
]

GENERIC_GREETING_PATTERNS = [
    r'\bdear (customer|user|member|client|sir|madam)\b',
]


def extract_urls(text: str):
    return URL_RE.findall(text or '')


def extract_email_addresses(text: str):
    return EMAIL_RE.findall(text or '')


def _domain(addr: str) -> str:
    addr = (addr or '').strip().lower()
    return addr.split('@', 1)[1] if '@' in addr else ''


def _find_matches(patterns, text: str):
    hits = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            value = re.sub(r'\s+', ' ', match.group(0).strip())
            hits.append(value)
    # preserve order, remove duplicates
    return list(dict.fromkeys(hits))


def _signal(key: str, title: str, description: str, score: float, matches=None, kind: str = 'risk'):
    return {
        'key': key,
        'title': title,
        'description': description,
        'score': float(score),
        'kind': kind,
        'matches': (matches or [])[:3],
    }


def _url_signal(urls):
    if not urls:
        return None

    risk = 0.0
    reasons = []

    for raw in urls[:5]:
        candidate = raw if raw.startswith(('http://', 'https://')) else f'http://{raw}'
        parsed = urlparse(candidate)
        host = (parsed.netloc or parsed.path).lower()

        if raw.lower().startswith('http://'):
            risk += 0.8
            reasons.append('uses insecure HTTP')

        if re.fullmatch(r'(?:\d{1,3}\.){3}\d{1,3}', host):
            risk += 1.1
            reasons.append('uses a raw IP address')

        if any(token in host for token in ['login', 'verify', 'secure', 'account', 'signin', 'update']):
            risk += 0.7
            reasons.append('contains login-style domain terms')

        if host.count('-') >= 2:
            risk += 0.3
            reasons.append('contains multiple hyphens')

    risk = min(risk, 2.0)
    if risk <= 0:
        return _signal(
            'link_present',
            'Link present',
            'The message contains a link, but no strongly suspicious link pattern was detected from the visible text alone.',
            0.0,
            matches=urls[:2],
            kind='context',
        )

    reasons_text = ', '.join(list(dict.fromkeys(reasons)))
    return _signal(
        'suspicious_link',
        'Suspicious link pattern',
        f'The message contains a link and the visible URL looks higher risk because it {reasons_text}.',
        risk,
        matches=urls[:2],
        kind='risk',
    )


def rule_based_evidence(text: str, sender_email: str = "", subject: str = ""):
    raw_text = text or ''
    text_l = re.sub(r'\s+', ' ', raw_text.lower()).strip()

    urls = extract_urls(raw_text)
    emails = extract_email_addresses(raw_text)

    risk_signals = []
    reassurance_signals = []
    context_signals = []
    sender_domain = _domain(sender_email)
    body_domains = sorted({_domain(e) for e in emails if _domain(e)})

    credential_hits = _find_matches(CREDENTIAL_REQUEST_PATTERNS, text_l)
    if credential_hits:
        risk_signals.append(_signal(
            'credential_request',
            'Credential request',
            'The message asks you to provide, confirm, or enter a password, code, or login detail.',
            2.4,
            credential_hits,
            'risk',
        ))

    security_hits = _find_matches(SECURITY_NOTIFICATION_PATTERNS, text_l)
    if security_hits:
        reassurance_signals.append(_signal(
            'security_notification',
            'Security notification context',
            'The wording looks more like a password-change notice or security alert than a direct request to hand over credentials.',
            -1.8,
            security_hits,
            'reassurance',
        ))

    urgency_hits = _find_matches(URGENCY_PATTERNS, text_l)
    if urgency_hits:
        risk_signals.append(_signal(
            'urgency',
            'Urgency pressure',
            'The message pushes the recipient to act quickly, which is a common phishing tactic.',
            1.1,
            urgency_hits,
            'risk',
        ))

    threat_hits = _find_matches(THREAT_PATTERNS, text_l)
    if threat_hits:
        risk_signals.append(_signal(
            'threat_language',
            'Threat language',
            'The message warns of negative consequences such as suspension, lockout, or account issues unless action is taken.',
            1.4,
            threat_hits,
            'risk',
        ))

    payment_hits = _find_matches(PAYMENT_PATTERNS, text_l)
    if payment_hits:
        risk_signals.append(_signal(
            'payment_language',
            'Payment or finance language',
            'The message discusses money or payments, which can raise phishing risk when combined with pressure or unusual requests.',
            0.8,
            payment_hits,
            'risk',
        ))

    generic_hits = _find_matches(GENERIC_GREETING_PATTERNS, text_l)
    if generic_hits:
        risk_signals.append(_signal(
            'generic_greeting',
            'Generic greeting',
            'The message uses a generic greeting rather than a named recipient.',
            0.5,
            generic_hits,
            'risk',
        ))

    if emails:
        context_signals.append(_signal(
            'email_address_present',
            'Contact address present',
            'The message includes an email address, which provides some contact context but does not by itself prove legitimacy.',
            0.0,
            emails[:2],
            'context',
        ))

    if sender_domain:
        context_signals.append(_signal(
            'sender_email_present',
            'Sender address available',
            'The sender email was provided and can be considered alongside the message content.',
            0.0,
            [sender_email],
            'context',
        ))

    if sender_domain and body_domains and all(d != sender_domain for d in body_domains):
        risk_signals.append(_signal(
            'domain_mismatch',
            'Domain mismatch',
            'The sender domain does not match the contact domain shown in the message body.',
            1.2,
            [f"sender={sender_domain}", f"body={', '.join(body_domains[:2])}"],
            'risk',
        ))

    link_signal = _url_signal(urls)
    if link_signal:
        target = risk_signals if link_signal['kind'] == 'risk' else context_signals
        target.append(link_signal)

    risk_score = round(sum(item['score'] for item in risk_signals) + sum(item['score'] for item in reassurance_signals), 2)

    return {
        'signals': risk_signals,
        'reassurance_signals': reassurance_signals,
        'context_signals': context_signals,
        'signal_titles': [s['title'] for s in risk_signals],
        'risk_score': risk_score,
        'url_count': len(urls),
        'urls': urls[:5],
        'email_addresses': emails[:5],
        'has_credential_request': any(s['key'] == 'credential_request' for s in risk_signals),
        'has_urgency': any(s['key'] == 'urgency' for s in risk_signals),
        'has_threat': any(s['key'] == 'threat_language' for s in risk_signals),
        'has_security_notification': any(s['key'] == 'security_notification' for s in reassurance_signals),
    }
