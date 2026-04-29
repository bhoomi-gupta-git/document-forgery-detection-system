"""
db/adapter.py — SQLite connection factory + CRUD helpers.

All database interaction in DocForge goes through this module.
Uses parameterised queries exclusively — no string formatting with user input.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager

import config

logger = logging.getLogger(__name__)


# ── Connection factory ────────────────────────────────────────────────────────

@contextmanager
def get_db():
    """Yield a sqlite3 connection with Row factory and FK enforcement."""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialisation ─────────────────────────────────────────────────────

def init_db():
    """Create tables and indexes from schema.sql. Safe to call on every startup."""
    try:
        with open(config.SCHEMA_PATH, 'r') as f:
            schema = f.read()
        with get_db() as conn:
            conn.executescript(schema)
        logger.info("Database schema initialised at %s", config.DB_PATH)
    except Exception as exc:
        logger.error("Failed to initialise database schema: %s", exc)
        raise


# ── documents CRUD ────────────────────────────────────────────────────────────

def insert_document(doc: dict) -> None:
    """
    Insert a new document record with status='pending'.

    Expected keys: id, filename_orig, filename_stored, file_ext,
                   file_size_bytes, mime_type, uploaded_at, status
    """
    sql = """
        INSERT INTO documents
            (id, filename_orig, filename_stored, file_ext,
             file_size_bytes, mime_type, uploaded_at, status)
        VALUES
            (:id, :filename_orig, :filename_stored, :file_ext,
             :file_size_bytes, :mime_type, :uploaded_at, :status)
    """
    with get_db() as conn:
        conn.execute(sql, doc)
    logger.info("Inserted document record id=%s status=%s", doc['id'], doc['status'])


def update_document_status(document_id: str, status: str) -> None:
    """Update the status column for a document. Valid values: pending|processing|complete|error."""
    with get_db() as conn:
        conn.execute(
            "UPDATE documents SET status = ? WHERE id = ?",
            (status, document_id)
        )
    logger.info("Document id=%s status → %s", document_id, status)


def get_document(document_id: str) -> sqlite3.Row | None:
    """Return a single documents row or None."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
    return row


# ── analysis_results CRUD ─────────────────────────────────────────────────────

def insert_result(result: dict) -> None:
    """
    Insert a completed analysis result.

    Expected keys: id, document_id, verdict, confidence, detections (list|None),
                   annotated_image, ocr_text, processing_ms, model_version,
                   analysed_at, error_message
    """
    # Serialise detections list → JSON string for storage
    detections_json = json.dumps(result.get('detections') or [])

    sql = """
        INSERT INTO analysis_results
            (id, document_id, verdict, confidence, detections,
             annotated_image, ocr_text, processing_ms, model_version,
             analysed_at, error_message)
        VALUES
            (:id, :document_id, :verdict, :confidence, :detections,
             :annotated_image, :ocr_text, :processing_ms, :model_version,
             :analysed_at, :error_message)
    """
    payload = dict(result)
    payload['detections'] = detections_json

    with get_db() as conn:
        conn.execute(sql, payload)
    logger.info(
        "Inserted result id=%s document_id=%s verdict=%s confidence=%.2f",
        result['id'], result['document_id'], result['verdict'], result['confidence']
    )


def get_result(document_id: str) -> dict | None:
    """
    Return a combined document + analysis_result dict for a given document UUID,
    or None if not found.
    """
    sql = """
        SELECT
            d.id              AS document_id,
            d.filename_orig,
            d.filename_stored,
            d.file_ext,
            d.file_size_bytes,
            d.mime_type,
            d.uploaded_at,
            d.status,
            r.id              AS result_id,
            r.verdict,
            r.confidence,
            r.detections,
            r.annotated_image,
            r.ocr_text,
            r.processing_ms,
            r.model_version,
            r.analysed_at,
            r.error_message
        FROM documents d
        LEFT JOIN analysis_results r ON r.document_id = d.id
        WHERE d.id = ?
    """
    with get_db() as conn:
        row = conn.execute(sql, (document_id,)).fetchone()

    if row is None:
        return None

    data = dict(row)
    # Deserialise detections JSON string → Python list
    raw_detections = data.get('detections')
    data['detections'] = json.loads(raw_detections) if raw_detections else []
    return data


def get_history(
    page: int = 1,
    limit: int = 10,
    verdict: str | None = None,
    sort: str = 'date',
    direction: str = 'desc',
    search: str | None = None,
) -> dict:
    """
    Return a paginated history list.

    Returns: {page, limit, total, items: [...]}
    """
    limit  = min(limit, config.HISTORY_PAGE_LIMIT_MAX)
    offset = (page - 1) * limit

    # Validate sort column against whitelist to prevent injection
    sort_col_map = {
        'date':       'd.uploaded_at',
        'filename':   'd.filename_orig',
        'verdict':    'r.verdict',
        'confidence': 'r.confidence',
    }
    sort_col = sort_col_map.get(sort, 'd.uploaded_at')
    sort_dir = 'DESC' if direction.lower() == 'desc' else 'ASC'

    # Build WHERE clauses
    where_clauses = []
    params: list = []

    if verdict and verdict in ('authentic', 'forged'):
        where_clauses.append("r.verdict = ?")
        params.append(verdict)

    if search:
        where_clauses.append("d.filename_orig LIKE ?")
        params.append(f"%{search}%")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    base_sql = f"""
        FROM documents d
        LEFT JOIN analysis_results r ON r.document_id = d.id
        {where_sql}
    """

    count_sql  = f"SELECT COUNT(*) {base_sql}"
    select_sql = f"""
        SELECT
            d.id              AS document_id,
            d.filename_orig,
            d.uploaded_at,
            d.status,
            r.id              AS result_id,
            r.verdict,
            r.confidence
        {base_sql}
        ORDER BY {sort_col} {sort_dir}
        LIMIT ? OFFSET ?
    """

    with get_db() as conn:
        total = conn.execute(count_sql, params).fetchone()[0]
        rows  = conn.execute(select_sql, params + [limit, offset]).fetchall()

    items = [dict(r) for r in rows]
    return {
        'page':  page,
        'limit': limit,
        'total': total,
        'items': items,
    }