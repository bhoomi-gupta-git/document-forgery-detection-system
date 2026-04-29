"""
utils/validator.py — File validation before the pipeline.

Validates by actual file bytes (python-magic), not browser claims.
Returns a ValidationResult namedtuple: (ok: bool, error: str | None)
"""

import logging
import os
from collections import namedtuple

import config

logger = logging.getLogger(__name__)

ValidationResult = namedtuple('ValidationResult', ['ok', 'error'])


def _get_mime(file_storage) -> str:
    """
    Detect MIME type from file bytes using python-magic.
    Falls back to stdlib mimetypes if libmagic is unavailable.
    """
    # Read first 2 KB for magic detection (don't consume the whole stream)
    header = file_storage.read(2048)
    file_storage.seek(0)                # always rewind after peeking

    try:
        import magic
        mime = magic.from_buffer(header, mime=True)
        return mime
    except ImportError:
        logger.warning("python-magic not available — falling back to mimetypes stdlib")
        import mimetypes
        filename = getattr(file_storage, 'filename', '')
        mime, _ = mimetypes.guess_type(filename)
        return mime or 'application/octet-stream'


def validate_upload(file_storage) -> ValidationResult:
    """
    Full validation gate for an uploaded FileStorage object.

    Checks (in order):
      1. File object is not None / empty filename
      2. File extension is in the whitelist
      3. File size is within the cap
      4. MIME type (by bytes) is in the whitelist

    Returns ValidationResult(ok=True, error=None) on success.
    Returns ValidationResult(ok=False, error='<friendly message>') on failure.
    """

    # ── 1. Presence check ─────────────────────────────────────────────────────
    if file_storage is None or file_storage.filename == '':
        return ValidationResult(False, "No file was provided.")

    filename = file_storage.filename
    ext = _get_extension(filename)

    # ── 2. Extension whitelist ────────────────────────────────────────────────
    if ext not in config.ALLOWED_EXTENSIONS:
        return ValidationResult(
            False,
            f"File type '.{ext}' is not supported. "
            f"Please upload a JPG, PNG, or PDF file."
        )

    # ── 3. Size check ─────────────────────────────────────────────────────────
    file_storage.seek(0, os.SEEK_END)
    size_bytes = file_storage.tell()
    file_storage.seek(0)

    if size_bytes > config.MAX_CONTENT_LENGTH:
        size_mb = size_bytes / (1024 * 1024)
        return ValidationResult(
            False,
            f"File size ({size_mb:.1f} MB) exceeds the {config.MAX_FILE_SIZE_MB} MB limit. "
            f"Please compress or re-scan the document."
        )

    if size_bytes == 0:
        return ValidationResult(False, "The uploaded file is empty.")

    # ── 4. MIME type (by bytes) ───────────────────────────────────────────────
    mime = _get_mime(file_storage)
    if mime not in config.ALLOWED_MIME_TYPES:
        logger.warning("MIME mismatch: filename=%s reported_mime=%s", filename, mime)
        return ValidationResult(
            False,
            "The file content does not match an accepted document type. "
            "Only JPG, PNG, and PDF files are accepted."
        )

    logger.debug(
        "Validation passed: filename=%s ext=%s mime=%s size=%d bytes",
        filename, ext, mime, size_bytes
    )
    return ValidationResult(True, None)


def _get_extension(filename: str) -> str:
    """Return lower-cased extension without the dot, e.g. 'pdf'."""
    _, ext = os.path.splitext(filename)
    return ext.lstrip('.').lower()


def get_file_metadata(file_storage) -> dict:
    """
    Return a dict of file metadata after successful validation.
    Call only after validate_upload() returns ok=True.
    """
    filename = file_storage.filename
    ext      = _get_extension(filename)
    mime     = _get_mime(file_storage)

    file_storage.seek(0, os.SEEK_END)
    size_bytes = file_storage.tell()
    file_storage.seek(0)

    return {
        'filename_orig':   filename,
        'file_ext':        'jpg' if ext == 'jpeg' else ext,
        'mime_type':       mime,
        'file_size_bytes': size_bytes,
    }