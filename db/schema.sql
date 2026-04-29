-- DocForge · Database Schema · v1.0
-- Single source of truth. Run once via db/adapter.py on startup.
-- Compatible with SQLite (V1) and PostgreSQL (V2 — change connection string only).

PRAGMA journal_mode = WAL;   -- better concurrent read performance
PRAGMA foreign_keys = ON;

-- ── Table: documents ─────────────────────────────────────────────────────────
-- One record per upload. Created when file is saved; status updated by pipeline.

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT        NOT NULL PRIMARY KEY,   -- UUID4 generated in Python
    filename_orig   TEXT        NOT NULL,               -- original name from user
    filename_stored TEXT        NOT NULL,               -- UUID-renamed name on disk
    file_ext        TEXT        NOT NULL,               -- jpg | png | pdf
    file_size_bytes INTEGER     NOT NULL,
    mime_type       TEXT        NOT NULL,               -- validated by python-magic
    uploaded_at     TEXT        NOT NULL,               -- UTC ISO-8601 string
    status          TEXT        NOT NULL                -- pending|processing|complete|error
                    CHECK (status IN ('pending','processing','complete','error'))
);

-- ── Table: analysis_results ──────────────────────────────────────────────────
-- One record per completed (or failed) analysis. FK → documents.

CREATE TABLE IF NOT EXISTS analysis_results (
    id                  TEXT    NOT NULL PRIMARY KEY,   -- UUID4
    document_id         TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    verdict             TEXT    NOT NULL                -- authentic|forged|unknown
                        CHECK (verdict IN ('authentic','forged','unknown')),
    confidence          REAL    NOT NULL,               -- 0.0 – 1.0 decimal
    detections          TEXT,                           -- JSON array (see TRD §4.4)
    annotated_image     TEXT,                           -- relative path in results/
    ocr_text            TEXT,                           -- full Tesseract string
    processing_ms       INTEGER,                        -- wall-clock pipeline time (ms)
    model_version       TEXT,                           -- semver e.g. '1.0.0'
    analysed_at         TEXT    NOT NULL,               -- UTC ISO-8601 string
    error_message       TEXT                            -- populated only when status=error
);

-- ── Table: users (stub for V2 auth — unused in V1) ───────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id              TEXT    NOT NULL PRIMARY KEY,
    email           TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at
    ON documents (uploaded_at DESC);                    -- history dashboard ordering

CREATE INDEX IF NOT EXISTS idx_results_document_id
    ON analysis_results (document_id);                  -- FK join performance

CREATE INDEX IF NOT EXISTS idx_results_verdict
    ON analysis_results (verdict);                      -- future verdict filtering