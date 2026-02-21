import re
from urllib.parse import urlparse

SUSPICIOUS_KEYWORDS = [
    ("credential_request", r"\b(password|passcode|otp|one[- ]time|verification code|login)\b"),
    ("urgent_language", r"\b(urgent|immediately|asap|within \d+ (minutes|hours)|act now)\b"),
    ("threat_language", r"\b(suspend|locked|terminated|unauthorized|security alert)\b"),
    ("money_language", r"\b(payment|invoice|bank|refund|wire|transfer|gift card)\b"),
    ("impersonation", r"\b(microsoft|paypal|amazon|apple|netflix|dhl|fedex|hmrc|royal mail)\b"),
]

def extract_urls(text: str):
    return re.findall(r"(https?://[^\s]+)", text)

def extract_email_addresses(text: str):
    return re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)

def rule_based_evidence(text: str):
    text_l = text.lower()

    hits = []
    for name, pattern in SUSPICIOUS_KEYWORDS:
        if re.search(pattern, text_l):
            hits.append(name)

    urls = extract_urls(text)
    emails = extract_email_addresses(text)

    return {
        "signals": hits,
        "url_count": len(urls),
        "urls": urls[:5],
        "email_addresses": emails[:5],
        "has_password_request": "credential_request" in hits,
        "has_urgency": "urgent_language" in hits,
        "has_threat": "threat_language" in hits,
    }

