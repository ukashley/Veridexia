from __future__ import annotations

import re
from urllib.parse import urlparse

INVISIBLE_TRANSLATION = str.maketrans('', '', '\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069\ufeff')
URL_RE = re.compile(r'hxxps?://\S+|https?://\S+|www\.\S+', re.I)
EMAIL_RE = re.compile(r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b', re.I)
PHONE_RE = re.compile(r'(?:\+?\d[\d\s().-]{7,}\d)')

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
    r'\bwe will never ask you to reply by email with your (password|card number|bank details?)\b',
    r'\bwe will never ask (?:you )?for (?:your )?(password|card number|bank details?)\b',
]

IDENTITY_VERIFICATION_PATTERNS = [
    r'\bverify your identity\b',
    r'\bidentity verification\b',
    r'\bupload your id\b',
    r'\bcomplete (?:an )?image verification\b',
    r'\bselfie or video confirmation\b',
    r'\bfinish your account set-?up\b',
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

TRANSACTIONAL_NOTIFICATION_PATTERNS = [
    r'\bsuccessfully collected payment\b',
    r'\bsuccessfully renewed\b',
    r'\border details below\b',
    r'\bamount charged\b',
    r'\brecurring reference number\b',
    r'\bmonthly plan\b',
    r'\brenewal\b',
    r'\bpayment due\b',
    r'\bpayment deadline\b',
    r'\belectricity account\b',
    r'\bbilling team\b',
    r'\bcustomer portal\b',
]

FORMAL_SERVICE_PATTERNS = [
    r'\bdear [a-z][a-z .-]{1,40},',
    r'\bwe are writing to\b',
    r'\bkind regards\b',
    r'\bcustomer support\b',
    r'\bwebsite:\s*(?:hxxps?://|https?://|www\.)',
]

NEWSLETTER_PATTERNS = [
    r'\bunsubscribe\b',
    r'\bmanage (?:your )?(?:email )?preferences\b',
    r'\bview (?:this )?email in (?:your )?browser\b',
    r'\bwhy did i get this email\b',
    r'\bmailing list\b',
    r'\bnewsletter\b',
]

JOB_ALERT_PATTERNS = [
    r'\bjob alerts?\b',
    r'\bopportunities and events\b',
    r'\bcareers? service\b',
    r'\bcareers? centre\b',
    r'\bvacanc(?:y|ies)\b',
    r'\bgraduate roles?\b',
    r'\bintern(?:ship)?\b',
]

RECIPIENT_EMAIL_PATTERNS = [
    r'(?:this message was sent to|this email was sent to|sent to)\s+([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})',
    r'(?:you are receiving this email at|you are receiving this message at)\s+([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})',
]

ACCOUNT_ADMINISTRATION_PATTERNS = [
    r'\bthis is confirmation that your .* account has (now )?been (deleted|closed|removed|deactivated)\b',
    r'\byou will no longer be able to recover your account\b',
    r'\bre-?register(?:ing)? (?:a )?new account\b',
    r'\baccount has (now )?been (deleted|closed|removed|deactivated)\b',
]

GENERIC_GREETING_PATTERNS = [
    r'\bdear (customer|user|member|client|sir|madam)\b',
]

GENERIC_BRAND_TOKENS = {
    'mail', 'email', 'teamtailor', 'noreply', 'reply', 'support', 'service',
    'help', 'alerts', 'notice', 'notify', 'news', 'info', 'com', 'co', 'uk',
    'org', 'net',
}


def extract_urls(text: str):
    return URL_RE.findall(text or '')


def extract_email_addresses(text: str):
    return EMAIL_RE.findall(text or '')


def _domain(addr: str) -> str:
    addr = (addr or '').strip().lower()
    return addr.split('@', 1)[1] if '@' in addr else ''


def _base_domain(host: str) -> str:
    host = (host or '').strip().lower()
    host = host.split(':', 1)[0].strip('.')
    if host.startswith('www.'):
        host = host[4:]
    labels = [part for part in host.split('.') if part]
    if len(labels) <= 2:
        return host
    if len(labels) >= 3 and '.'.join(labels[-2:]) in {'co.uk', 'ac.uk', 'gov.uk', 'org.uk'}:
        return '.'.join(labels[-3:])
    return '.'.join(labels[-2:])


def _url_host(raw: str) -> str:
    value = (raw or '').strip().rstrip(').,;]')
    value = re.sub(r'^hxxp', 'http', value, flags=re.I)
    candidate = value if re.match(r'https?://', value, flags=re.I) else f'http://{value}'
    parsed = urlparse(candidate)
    return (parsed.netloc or parsed.path).lower().split('/', 1)[0]


def _url_domains(urls):
    domains = []
    for raw in urls or []:
        host = _url_host(raw)
        if host:
            domains.append(_base_domain(host))
    return sorted({domain for domain in domains if domain})


def _brand_tokens(addr: str):
    domain = _domain(addr)
    tokens = []
    for label in domain.split('.'):
        for token in re.split(r'[^a-z0-9]+', label):
            if len(token) < 4 or token.isdigit() or token in GENERIC_BRAND_TOKENS:
                continue
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def _compact_text(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', (text or '').lower())


def _normalize_text(text: str) -> str:
    cleaned = (text or '').translate(INVISIBLE_TRANSLATION).replace('\u00a0', ' ')
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    cleaned = re.sub(r' ?\n ?', '\n', cleaned)
    return cleaned.strip()


def _find_matches(patterns, text: str):
    hits = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            value = re.sub(r'\s+', ' ', match.group(0).strip())
            hits.append(value)
    # Keep the first natural-looking examples so the UI can show snippets that still read like the source email.
    return list(dict.fromkeys(hits))


def _recipient_domains(text: str):
    hits = []
    for pattern in RECIPIENT_EMAIL_PATTERNS:
        for match in re.finditer(pattern, text or '', flags=re.I | re.S):
            hits.append(_domain(match.group(1)))
    return {domain for domain in hits if domain}


def _has_phone_number(text: str) -> bool:
    return bool(PHONE_RE.search(text or ''))


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

    # Most legitimate emails include links, so we only escalate when the visible URL
    # also carries patterns that tend to look suspicious on their own.
    for raw in urls[:5]:
        host = _url_host(raw)

        if raw.lower().startswith('http://') or raw.lower().startswith('hxxp://'):
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


def _sender_domain_signal(sender_domain: str):
    if not sender_domain:
        return None

    risk = 0.0
    reasons = []
    host = sender_domain.lower()

    if any(token in host for token in ['login', 'verify', 'secure', 'reset', 'update', 'account']):
        risk += 0.7
        reasons.append('contains phishing-style sender terms')

    if host.count('-') >= 2:
        risk += 0.3
        reasons.append('contains multiple hyphens')

    if risk <= 0:
        return None

    reasons_text = ', '.join(list(dict.fromkeys(reasons)))
    return _signal(
        'suspicious_sender_domain',
        'Suspicious sender domain',
        f'The sender domain looks less trustworthy because it {reasons_text}.',
        min(risk, 1.6),
        matches=[sender_domain],
        kind='risk',
    )


def rule_based_evidence(text: str, sender_email: str = "", subject: str = ""):
    raw_text = _normalize_text(text)
    text_l = re.sub(r'\s+', ' ', raw_text.lower()).strip()
    subject_l = re.sub(r'\s+', ' ', _normalize_text(subject).lower()).strip()
    combined_l = f'{subject_l} {text_l}'.strip()
    combined_compact = _compact_text(combined_l)

    # This layer is intentionally not "phishing cues only".
    # It also looks for reassuring patterns so the app can explain borderline but routine emails more fairly.
    urls = extract_urls(raw_text)
    emails = extract_email_addresses(raw_text)

    risk_signals = []
    reassurance_signals = []
    context_signals = []
    sender_domain = _domain(sender_email)
    sender_base_domain = _base_domain(sender_domain)
    url_domains = _url_domains(urls)
    recipient_domains = _recipient_domains(raw_text)
    body_domains = sorted({_domain(e) for e in emails if _domain(e) and _domain(e) not in recipient_domains})

    credential_hits = _find_matches(CREDENTIAL_REQUEST_PATTERNS, combined_l)
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

    identity_hits = _find_matches(IDENTITY_VERIFICATION_PATTERNS, combined_l)
    if identity_hits:
        reassurance_signals.append(_signal(
            'identity_verification_context',
            'Identity verification context',
            'The wording looks like a routine identity or KYC verification flow rather than a password or code request.',
            -1.4,
            identity_hits,
            'reassurance',
        ))

    urgency_hits = _find_matches(URGENCY_PATTERNS, combined_l)
    if urgency_hits:
        risk_signals.append(_signal(
            'urgency',
            'Urgency pressure',
            'The message pushes the recipient to act quickly, which is a common phishing tactic.',
            1.1,
            urgency_hits,
            'risk',
        ))

    threat_hits = _find_matches(THREAT_PATTERNS, combined_l)
    if threat_hits:
        risk_signals.append(_signal(
            'threat_language',
            'Threat language',
            'The message warns of negative consequences such as suspension, lockout, or account issues unless action is taken.',
            1.4,
            threat_hits,
            'risk',
        ))

    payment_hits = _find_matches(PAYMENT_PATTERNS, combined_l)
    if payment_hits:
        risk_signals.append(_signal(
            'payment_language',
            'Payment or finance language',
            'The message discusses money or payments, which can raise phishing risk when combined with pressure or unusual requests.',
            0.4,
            payment_hits,
            'risk',
        ))

    transactional_hits = _find_matches(TRANSACTIONAL_NOTIFICATION_PATTERNS, combined_l)
    if transactional_hits:
        reassurance_signals.append(_signal(
            'transactional_notification',
            'Routine billing or service notice',
            'The message reads like a routine billing, renewal, or order-confirmation notice rather than a request for credentials or urgent action.',
            -0.9,
            transactional_hits,
            'reassurance',
        ))

    formal_hits = _find_matches(FORMAL_SERVICE_PATTERNS, raw_text.lower())
    if formal_hits:
        reassurance_signals.append(_signal(
            'formal_service_message',
            'Formal service wording',
            'The message uses formal service-style wording and includes normal customer-support context.',
            -0.6,
            formal_hits,
            'reassurance',
        ))

    if _has_phone_number(raw_text) and re.search(r'\b(customer support|support|contact)\b', combined_l):
        reassurance_signals.append(_signal(
            'support_contact_present',
            'Support contact present',
            'The message includes customer-support contact details, which is useful context for manual verification.',
            -0.4,
            PHONE_RE.findall(raw_text)[:2],
            'reassurance',
        ))

    newsletter_hits = _find_matches(NEWSLETTER_PATTERNS, combined_l)
    if newsletter_hits:
        reassurance_signals.append(_signal(
            'newsletter_context',
            'Newsletter or mailing-list context',
            'The message contains typical newsletter or mailing-list wording, which often appears in legitimate promotional or update emails.',
            -0.8,
            newsletter_hits,
            'reassurance',
        ))

    job_alert_hits = _find_matches(JOB_ALERT_PATTERNS, combined_l)
    if job_alert_hits:
        reassurance_signals.append(_signal(
            'job_alert_context',
            'Job alert or careers context',
            'The message reads like a careers update, vacancy alert, or opportunities bulletin rather than a credential-harvesting message.',
            -0.9,
            job_alert_hits,
            'reassurance',
        ))

    account_admin_hits = _find_matches(ACCOUNT_ADMINISTRATION_PATTERNS, combined_l)
    if account_admin_hits:
        reassurance_signals.append(_signal(
            'account_administration_notice',
            'Account administration notice',
            'The message reads more like an administrative account update or confirmation than a request for credentials or urgent action.',
            -1.0,
            account_admin_hits,
            'reassurance',
        ))

    generic_hits = _find_matches(GENERIC_GREETING_PATTERNS, combined_l)
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

        sender_domain_signal = _sender_domain_signal(sender_domain)
        if sender_domain_signal:
            risk_signals.append(sender_domain_signal)

    sender_brand_hits = []
    for token in _brand_tokens(sender_email):
        compact_token = _compact_text(token)
        if compact_token and compact_token in combined_compact:
            sender_brand_hits.append(token)
    if sender_brand_hits:
        reassurance_signals.append(_signal(
            'sender_brand_match',
            'Sender brand match',
            'The sender brand appears in the subject or message content.',
            -0.5,
            sender_brand_hits[:2],
            'reassurance',
        ))

    if sender_domain and sender_domain in body_domains:
        reassurance_signals.append(_signal(
            'domain_match',
            'Domain match',
            'The sender domain matches the contact domain shown in the message body.',
            -0.6,
            [f"sender={sender_domain}", f"body={sender_domain}"],
            'reassurance',
        ))

    if sender_base_domain and url_domains and sender_base_domain in url_domains:
        reassurance_signals.append(_signal(
            'link_domain_match',
            'Link domain matches sender',
            'The visible link domain appears to match the sender organisation domain.',
            -1.2,
            [f"sender={sender_base_domain}", f"link={sender_base_domain}"],
            'reassurance',
        ))

    if sender_base_domain and any(
        re.search(rf'\b(?:hxxps?://|https?://|www\.)[^\s]+{re.escape(sender_base_domain)}', raw_text, flags=re.I)
        and re.search(r'\b(website|official website|customer portal)\b', combined_l)
        for _ in [0]
    ):
        reassurance_signals.append(_signal(
            'official_website_present',
            'Official website or portal present',
            'The message includes an official website or customer portal that matches the sender organisation.',
            -0.7,
            [sender_base_domain],
            'reassurance',
        ))

    if security_hits and not credential_hits:
        reassurance_signals.append(_signal(
            'no_email_credential_request',
            'No email credential request',
            'The message explicitly says not to reply by email with passwords, card numbers, or bank details.',
            -1.0,
            security_hits,
            'reassurance',
        ))

    if sender_domain and body_domains and all(_base_domain(d) != sender_base_domain for d in body_domains):
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
