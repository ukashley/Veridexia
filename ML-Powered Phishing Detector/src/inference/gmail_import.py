from __future__ import annotations

import base64
from dataclasses import dataclass, field
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime, parseaddr
from pathlib import Path

from bs4 import BeautifulSoup


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
MAX_EMAIL_BODY = 12000


@dataclass
class GmailMessage:
    message_id: str
    subject: str = ''
    sender_email: str = ''
    sender_label: str = ''
    body_text: str = ''
    snippet: str = ''
    received_at: str = ''
    body_loaded: bool = False


@dataclass
class GmailImportResult:
    items: list[GmailMessage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ''
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _decode_body_data(value: str | None) -> str:
    if not value:
        return ''
    try:
        raw = base64.urlsafe_b64decode(value.encode('utf-8'))
    except Exception:
        return ''

    for encoding in ('utf-8', 'latin-1'):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='ignore')


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or '', 'html.parser')
    return soup.get_text('\n', strip=True)


def _clean_text(text: str, *, limit: int = MAX_EMAIL_BODY) -> str:
    clean = (text or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + '\n...[truncated]'


def _extract_text_from_payload(payload: dict | None) -> str:
    if not payload:
        return ''

    plain_parts: list[str] = []
    html_parts: list[str] = []
    queue = [payload]

    while queue:
        current = queue.pop(0)
        mime_type = (current.get('mimeType') or '').lower()
        body = current.get('body') or {}
        parts = current.get('parts') or []
        data = body.get('data')

        if data:
            decoded = _decode_body_data(data)
            if decoded.strip():
                if mime_type == 'text/html':
                    html_parts.append(_html_to_text(decoded))
                else:
                    plain_parts.append(decoded)

        queue.extend(parts)

    # Prefer plain text when Gmail gives both plain and HTML versions.
    # It is usually cleaner and closer to what the models were trained on.
    if plain_parts:
        return _clean_text('\n\n'.join(part for part in plain_parts if part.strip()))
    if html_parts:
        return _clean_text('\n\n'.join(part for part in html_parts if part.strip()))
    return ''


def _format_received_at(value: str | None) -> str:
    if not value:
        return ''

    try:
        return parsedate_to_datetime(value).strftime('%Y-%m-%d %H:%M')
    except Exception:
        return value


def _header_map(payload: dict | None) -> dict[str, str]:
    headers = {}
    for item in (payload or {}).get('headers', []):
        name = (item.get('name') or '').strip().lower()
        if name:
            headers[name] = _decode_header_value(item.get('value'))
    return headers


def _build_message(
    message_id: str,
    payload: dict,
    snippet: str = '',
    *,
    include_body: bool = True,
) -> GmailMessage:
    headers = _header_map(payload)
    sender_label = headers.get('from', '')
    sender_email = parseaddr(sender_label)[1]
    clean_snippet = _clean_text(snippet, limit=300)

    return GmailMessage(
        message_id=message_id,
        subject=headers.get('subject', '(no subject)') or '(no subject)',
        sender_email=sender_email,
        sender_label=sender_label or sender_email,
        body_text=_extract_text_from_payload(payload) if include_body else '',
        snippet=clean_snippet,
        received_at=_format_received_at(headers.get('date')),
        body_loaded=include_body,
    )


def _load_gmail_dependencies():
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise RuntimeError(
            'Google API packages are not installed. Add google-api-python-client, '
            'google-auth-httplib2, and google-auth-oauthlib to requirements.'
        ) from exc

    return Request, Credentials, InstalledAppFlow, build


def _get_gmail_service(credentials_path: Path, token_path: Path):
    warnings: list[str] = []
    try:
        Request, Credentials, InstalledAppFlow, build = _load_gmail_dependencies()
    except RuntimeError as exc:
        return None, [str(exc)]

    if not credentials_path.exists():
        return None, [f'Google OAuth credentials were not found at {credentials_path.name}.']

    creds = None
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            warnings.append('Existing token.json could not be read and will be replaced after sign-in.')

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json(), encoding='utf-8')

    try:
        service = build('gmail', 'v1', credentials=creds, cache_discovery=False)
    except Exception as exc:
        warnings.append(f'Gmail import failed: {exc}')
        return None, warnings

    return service, warnings


def _fetch_message_previews(service, message_ids: list[str]):
    previews: dict[int, GmailMessage] = {}
    warnings: list[str] = []

    def callback(request_id, response, exception):
        index = int(request_id)
        if exception is not None:
            warnings.append(f'Could not import one Gmail message preview: {exception}')
            return
        previews[index] = _build_message(
            message_ids[index],
            response.get('payload') or {},
            snippet=response.get('snippet', ''),
            include_body=False,
        )

    batch = service.new_batch_http_request(callback=callback)
    for index, message_id in enumerate(message_ids):
        batch.add(
            service.users().messages().get(
                userId='me',
                id=message_id,
                format='metadata',
                metadataHeaders=['Subject', 'From', 'Date'],
            ),
            request_id=str(index),
        )

    batch.execute()

    ordered = [previews[index] for index in range(len(message_ids)) if index in previews]
    return ordered, warnings


def import_recent_gmail_messages(
    credentials_path: Path,
    token_path: Path,
    *,
    max_results: int = 10,
) -> GmailImportResult:
    result = GmailImportResult()

    service, warnings = _get_gmail_service(credentials_path, token_path)
    result.warnings.extend(warnings)
    if service is None:
        return result

    try:
        response = service.users().messages().list(
            userId='me',
            labelIds=['INBOX'],
            maxResults=max(1, int(max_results)),
        ).execute()
    except Exception as exc:
        result.warnings.append(f'Gmail import failed: {exc}')
        return result

    messages = response.get('messages', [])
    if not messages:
        result.warnings.append('No inbox messages were returned by Gmail.')
        return result

    message_ids = [item['id'] for item in messages if item.get('id')]
    previews, preview_warnings = _fetch_message_previews(service, message_ids)
    result.items.extend(previews)
    result.warnings.extend(preview_warnings)

    return result


def load_gmail_message_body(
    credentials_path: Path,
    token_path: Path,
    message_id: str,
):
    service, warnings = _get_gmail_service(credentials_path, token_path)
    if service is None:
        return None, warnings

    try:
        details = service.users().messages().get(
            userId='me',
            id=message_id,
            format='full',
        ).execute()
    except Exception as exc:
        warnings.append(f'Could not load the selected Gmail message: {exc}')
        return None, warnings

    message = _build_message(
        message_id,
        details.get('payload') or {},
        snippet=details.get('snippet', ''),
        include_body=True,
    )
    return message, warnings
