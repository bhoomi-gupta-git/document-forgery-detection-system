"""
conftest.py — Pytest fixtures for DocForge test suite.

Provides:
  - app: Flask test application instance (in-memory SQLite, temp dirs)
  - client: Flask test client
  - db_conn: raw sqlite3 connection to the test database
  - sample_jpeg: minimal valid JPEG bytes
  - sample_png: minimal valid PNG bytes
  - sample_pdf: minimal valid PDF bytes
  - tmp_upload_dir / tmp_results_dir: temp filesystem directories
"""

import io
import os
import struct
import tempfile
import sqlite3
import zlib

import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal valid file bytes
# ---------------------------------------------------------------------------

def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Return bytes of a minimal valid JPEG (solid white, width×height)."""
    # Use a pre-built minimal JPEG header for a 1×1 white pixel,
    # then let the test scale expectations accordingly.
    # For simplicity we embed a hardcoded tiny JPEG.
    MINIMAL_JPEG = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
        b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00"
        b"\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b"
        b"\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04"
        b"\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa"
        b"\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n"
        b"\x16\x17\x18\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZ"
        b"cdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95"
        b"\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3"
        b"\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca"
        b"\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7"
        b"\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa"
        b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd3P\x00\x00\x00\x1f\xff\xd9"
    )
    return MINIMAL_JPEG


def _make_png(width: int = 64, height: int = 64) -> bytes:
    """Return bytes of a minimal valid PNG (solid white, width×height)."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # Build raw image data: each row = filter_byte(0) + RGB pixels
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00" + (b"\xff\xff\xff" * width)
    compressed = zlib.compress(raw_rows)
    idat = _chunk(b"IDAT", compressed)
    iend = _chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


def _make_pdf() -> bytes:
    """Return bytes of a minimal valid single-page PDF."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] >>\nendobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer\n<< /Size 4 /Root 1 0 R >>\n"
        b"startxref\n190\n%%EOF\n"
    )


# ---------------------------------------------------------------------------
# App & client fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_dirs():
    """Create temporary upload and results directories for the test session."""
    with tempfile.TemporaryDirectory() as upload_dir, \
         tempfile.TemporaryDirectory() as results_dir:
        yield {"upload": upload_dir, "results": results_dir}


@pytest.fixture(scope="session")
def app(tmp_dirs):
    """
    Create and configure a Flask app instance for testing.

    Uses:
      - In-memory SQLite database
      - Temporary upload/results directories
      - TESTING=True, no real ML model required
    """
    # Import here so conftest doesn't fail if dependencies aren't installed yet
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # Patch config before importing app
    import config
    config.UPLOAD_FOLDER = tmp_dirs["upload"]
    config.RESULTS_FOLDER = tmp_dirs["results"]
    config.DATABASE = ":memory:"
    config.MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    config.ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "pdf"}
    config.ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "application/pdf"}
    config.MODEL_PATH = None  # forces mock/fallback in detector.py

    from app import create_app
    flask_app = create_app()
    flask_app.config.update(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "MAX_CONTENT_LENGTH": 10 * 1024 * 1024,
        }
    )

    # Initialise schema on the in-memory db
    with flask_app.app_context():
        from db.adapter import init_db
        init_db()

    yield flask_app


@pytest.fixture()
def client(app):
    """Return a Flask test client."""
    with app.test_client() as c:
        yield c


@pytest.fixture()
def app_context(app):
    """Push an application context for tests that need it."""
    with app.app_context():
        yield


# ---------------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def db_conn(app):
    """
    Yield a raw sqlite3 connection to the test database.
    Rolls back any changes after each test to keep tests isolated.
    """
    import config as cfg
    conn = sqlite3.connect(cfg.DATABASE)
    conn.row_factory = sqlite3.Row
    yield conn
    conn.rollback()
    conn.close()


# ---------------------------------------------------------------------------
# File byte fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_jpeg() -> bytes:
    """Return bytes of a minimal valid JPEG."""
    return _make_jpeg()


@pytest.fixture(scope="session")
def sample_png() -> bytes:
    """Return bytes of a minimal valid PNG (64×64, solid white)."""
    return _make_png()


@pytest.fixture(scope="session")
def sample_pdf() -> bytes:
    """Return bytes of a minimal valid single-page PDF."""
    return _make_pdf()


@pytest.fixture(scope="session")
def oversized_bytes() -> bytes:
    """Return bytes that exceed the 10 MB upload limit."""
    return b"A" * (10 * 1024 * 1024 + 1)


@pytest.fixture(scope="session")
def corrupt_bytes() -> bytes:
    """Return bytes that look like a JPEG header but are truncated/corrupt."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 20 + b"CORRUPT_DATA"


@pytest.fixture(scope="session")
def wrong_type_bytes() -> bytes:
    """Return bytes of a plain text file (disallowed type)."""
    return b"Hello, this is a plain text file, not a document image."


# ---------------------------------------------------------------------------
# FileStorage helpers for multipart uploads
# ---------------------------------------------------------------------------

@pytest.fixture()
def jpeg_upload(sample_jpeg):
    """Return a (BytesIO, filename, content_type) tuple for a JPEG upload."""
    return (io.BytesIO(sample_jpeg), "test_document.jpg", "image/jpeg")


@pytest.fixture()
def png_upload(sample_png):
    """Return a (BytesIO, filename, content_type) tuple for a PNG upload."""
    return (io.BytesIO(sample_png), "test_document.png", "image/png")


@pytest.fixture()
def pdf_upload(sample_pdf):
    """Return a (BytesIO, filename, content_type) tuple for a PDF upload."""
    return (io.BytesIO(sample_pdf), "test_document.pdf", "application/pdf")


# ---------------------------------------------------------------------------
# Pre-seeded DB record fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_result(app, tmp_dirs):
    """
    Insert a complete analysis result record into the test DB.
    Returns the document UUID so tests can reference it via GET /api/result/<id>.
    """
    import uuid
    from datetime import datetime

    doc_id = str(uuid.uuid4())
    result_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    import config as cfg
    conn = sqlite3.connect(cfg.DATABASE)
    try:
        conn.execute(
            """
            INSERT INTO documents
              (id, filename_orig, filename_stored, file_ext,
               file_size_bytes, mime_type, uploaded_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, "seed_doc.jpg", f"{doc_id}.jpg", "jpg",
             1024, "image/jpeg", now, "complete"),
        )
        conn.execute(
            """
            INSERT INTO analysis_results
              (id, document_id, verdict, confidence, detections,
               annotated_image, ocr_text, processing_ms,
               model_version, analysed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (result_id, doc_id, "authentic", 0.92,
             "[]", None, "Sample OCR text", 4200, "1.0.0", now),
        )
        conn.commit()
    finally:
        conn.close()

    return {"doc_id": doc_id, "result_id": result_id}


# ---------------------------------------------------------------------------
# Utility: post a file to /api/upload via test client
# ---------------------------------------------------------------------------

def post_file(client, file_bytes: bytes, filename: str, content_type: str):
    """Helper to POST a file to /api/upload using the Flask test client."""
    return client.post(
        "/api/upload",
        data={"file": (io.BytesIO(file_bytes), filename)},
        content_type="multipart/form-data",
    )