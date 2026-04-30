from __future__ import annotations

from dataclasses import dataclass, field
from email import policy
from email.header import decode_header, make_header
from email.parser import BytesParser
from email.utils import parseaddr
from io import BytesIO
import json
import re
import zipfile
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


MAX_TEXT_PER_FILE = 12000
MAX_COMBINED_TEXT = 24000
TEXT_EXTENSIONS = {
    '.txt', '.text', '.md', '.rst', '.log', '.csv', '.tsv', '.json',
    '.yaml', '.yml', '.ini', '.cfg', '.html', '.htm', '.xml',
}


@dataclass
class ExtractedUpload:
    filename: str
    kind: str
    text: str = ''
    subject: str = ''
    sender_email: str = ''
    warnings: list[str] = field(default_factory=list)


@dataclass
class UploadContext:
    body_text: str = ''
    subject: str = ''
    sender_email: str = ''
    items: list[ExtractedUpload] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _decode_text_bytes(data: bytes) -> str:
    for encoding in ('utf-8', 'utf-16', 'latin-1'):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='ignore')


def _clip_text(text: str, *, limit: int = MAX_TEXT_PER_FILE) -> tuple[str, bool]:
    clean = re.sub(r'\r\n?', '\n', text or '').strip()
    if len(clean) <= limit:
        return clean, False
    return clean[:limit].rstrip() + '\n...[truncated]', True


def _decode_header_value(value: str | None) -> str:
    if not value:
        return ''
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return value.strip()


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or '', 'html.parser')
    return soup.get_text('\n', strip=True)


def _extract_docx_text(data: bytes) -> str:
    with zipfile.ZipFile(BytesIO(data)) as zf:
        document_xml = zf.read('word/document.xml')

    root = ET.fromstring(document_xml)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    paragraphs = []
    for paragraph in root.findall('.//w:p', ns):
        parts = []
        for node in paragraph.findall('.//w:t', ns):
            if node.text:
                parts.append(node.text)
        line = ''.join(parts).strip()
        if line:
            paragraphs.append(line)
    return '\n'.join(paragraphs)


def _extract_pdf_text(data: bytes) -> tuple[str, list[str]]:
    warnings = []
    if PdfReader is None:
        warnings.append('PDF text extraction is unavailable. Install pypdf to scan PDF attachments.')
        return '', warnings

    try:
        reader = PdfReader(BytesIO(data))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ''
            if page_text.strip():
                pages.append(page_text.strip())
        if not pages:
            warnings.append('No readable text was found in the PDF.')
        return '\n\n'.join(pages), warnings
    except Exception:
        warnings.append('PDF text extraction failed for this file.')
        return '', warnings


def _flatten_json(value, prefix: str = '') -> list[str]:
    lines = []
    if isinstance(value, dict):
        for key, item in value.items():
            label = f'{prefix}{key}'
            if isinstance(item, (dict, list)):
                lines.extend(_flatten_json(item, prefix=label + '.'))
            else:
                lines.append(f'{label}: {item}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            label = f'{prefix}{index}'
            if isinstance(item, (dict, list)):
                lines.extend(_flatten_json(item, prefix=label + '.'))
            else:
                lines.append(f'{label}: {item}')
    else:
        lines.append(f'{prefix.rstrip(".")}: {value}')
    return lines


def _extract_text_attachment(part, filename: str, depth: int) -> ExtractedUpload:
    payload = part.get_payload(decode=True) or b''
    return extract_uploaded_file(
        filename,
        payload,
        content_type=part.get_content_type(),
        depth=depth + 1,
    )


def _extract_eml_text(data: bytes, depth: int) -> ExtractedUpload:
    msg = BytesParser(policy=policy.default).parsebytes(data)
    subject = _decode_header_value(msg.get('subject'))
    sender_email = parseaddr(_decode_header_value(msg.get('from')))[1]
    sections = []
    warnings = []

    for part in msg.walk():
        content_type = part.get_content_type()
        disposition = part.get_content_disposition()
        filename = _decode_header_value(part.get_filename())

        if disposition != 'attachment':
            try:
                payload = part.get_content()
            except Exception:
                payload = ''

            if content_type == 'text/plain' and isinstance(payload, str) and payload.strip():
                sections.append(payload.strip())
            elif content_type == 'text/html' and isinstance(payload, str):
                html_text = _html_to_text(payload)
                if html_text:
                    sections.append(html_text)
            continue

        # Follow text-like attachments inside uploaded emails, but stop after a couple of levels
        # so a messy .eml file cannot blow up scan time.
        if not filename or depth >= 2:
            continue

        nested = _extract_text_attachment(part, filename, depth)
        if nested.text:
            sections.append(f'[Attachment: {nested.filename}]\n{nested.text}')
        warnings.extend(nested.warnings)

    text = '\n\n'.join(section for section in sections if section).strip()
    clipped, was_clipped = _clip_text(text)
    if was_clipped:
        warnings.append(f'Email content extracted from attachments was truncated for {subject or "uploaded email"}.')

    return ExtractedUpload(
        filename='email.eml',
        kind='email',
        text=clipped,
        subject=subject,
        sender_email=sender_email,
        warnings=warnings,
    )


def extract_uploaded_file(filename: str, data: bytes, content_type: str = '', depth: int = 0) -> ExtractedUpload:
    suffix = ''
    if filename:
        lowered = filename.lower()
        suffix = lowered[lowered.rfind('.'):] if '.' in lowered else ''

    warnings: list[str] = []
    extracted_text = ''
    kind = 'file'
    subject = ''
    sender_email = ''

    if suffix in {'.txt', '.text', '.md', '.rst', '.log', '.csv', '.tsv', '.yaml', '.yml', '.ini', '.cfg', '.xml'}:
        kind = 'text'
        extracted_text = _decode_text_bytes(data)
    elif suffix == '.json':
        kind = 'json'
        try:
            parsed = json.loads(_decode_text_bytes(data))
            extracted_text = '\n'.join(_flatten_json(parsed))
        except Exception:
            extracted_text = _decode_text_bytes(data)
    elif suffix in {'.html', '.htm'} or content_type == 'text/html':
        kind = 'html'
        extracted_text = _html_to_text(_decode_text_bytes(data))
    elif suffix == '.docx':
        kind = 'docx'
        try:
            extracted_text = _extract_docx_text(data)
        except Exception:
            warnings.append('DOCX text extraction failed for this file.')
    elif suffix == '.pdf':
        kind = 'pdf'
        extracted_text, pdf_warnings = _extract_pdf_text(data)
        warnings.extend(pdf_warnings)
    elif suffix == '.eml' or content_type == 'message/rfc822':
        nested = _extract_eml_text(data, depth)
        nested.filename = filename or nested.filename
        return nested
    else:
        is_text_like = (
            content_type.startswith('text/')
            or suffix in TEXT_EXTENSIONS
        )
        if is_text_like:
            kind = 'text'
            extracted_text = _decode_text_bytes(data)
        else:
            warnings.append(f'{filename or "Uploaded file"} is not a supported extractable format yet.')

    clipped_text, was_clipped = _clip_text(extracted_text)
    if was_clipped:
        warnings.append(f'Extracted text from {filename or "uploaded file"} was truncated to keep scanning responsive.')

    return ExtractedUpload(
        filename=filename or 'uploaded-file',
        kind=kind,
        text=clipped_text,
        subject=subject,
        sender_email=sender_email,
        warnings=warnings,
    )


def build_upload_context(uploaded_files) -> UploadContext:
    items: list[ExtractedUpload] = []
    warnings: list[str] = []
    body_parts: list[str] = []
    subject = ''
    sender_email = ''

    for uploaded_file in uploaded_files or []:
        item = extract_uploaded_file(
            getattr(uploaded_file, 'name', 'uploaded-file'),
            uploaded_file.getvalue(),
            content_type=getattr(uploaded_file, 'type', '') or '',
        )
        items.append(item)
        warnings.extend(item.warnings)

        if item.subject and not subject:
            subject = item.subject
        if item.sender_email and not sender_email:
            sender_email = item.sender_email
        if item.text:
            body_parts.append(f'[File: {item.filename}]\n{item.text}')

    # Uploaded text is merged into one scan body so pasted content and file content
    # can be analysed together as a single message context.
    body_text = '\n\n'.join(part for part in body_parts if part).strip()
    clipped_body, was_clipped = _clip_text(body_text, limit=MAX_COMBINED_TEXT)
    if was_clipped:
        warnings.append('Combined upload text was truncated before analysis.')

    return UploadContext(
        body_text=clipped_body,
        subject=subject,
        sender_email=sender_email,
        items=items,
        warnings=warnings,
    )
