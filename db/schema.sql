-- DocForge · Database Schema · v1.1
-- Updated for multi-page PDF support.
-- New columns: total_pages, pages_summary, forged_pages

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Table: documents ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT        NOT NULL PRIMARY KEY,
    filename_orig   TEXT        NOT NULL,
    filename_stored TEXT        NOT NULL,
    file_ext        TEXT        NOT NULL,
    file_size_bytes INTEGER     NOT NULL,
    mime_type       TEXT        NOT NULL,
    uploaded_at     TEXT        NOT NULL,
    status          TEXT        NOT NULL
                    CHECK (status IN ('pending','processing','complete','error'))
);

-- ── Table: analysis_results ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS analysis_results (
    id                  TEXT    NOT NULL PRIMARY KEY,
    document_id         TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    verdict             TEXT    NOT NULL
                        CHECK (verdict IN ('authentic','forged','unknown')),
    confidence          REAL    NOT NULL,
    detections          TEXT,                   -- JSON array
    annotated_image     TEXT,                   -- best page heatmap filename
    ocr_text            TEXT,
    processing_ms       INTEGER,
    model_version       TEXT,
    analysed_at         TEXT    NOT NULL,
    error_message       TEXT,
    -- ── Multi-page fields ─────────────────────────────────────────────────
    total_pages         INTEGER DEFAULT 1,      -- total pages in document
    pages_summary       TEXT,                   -- JSON array of per-page results
    forged_pages        TEXT                    -- JSON array of forged page numbers
);

-- ── Table: users (stub for V2) ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    id              TEXT    NOT NULL PRIMARY KEY,
    email           TEXT    NOT NULL UNIQUE,
    password_hash   TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_documents_uploaded_at
    ON documents (uploaded_at DESC);

CREATE INDEX IF NOT EXISTS idx_results_document_id
    ON analysis_results (document_id);

CREATE INDEX IF NOT EXISTS idx_results_verdict
    ON analysis_results (verdict);

-- ── Migration: add multi-page columns if upgrading from v1.0 ─────────────────
-- These are safe to run on existing databases — ignored if columns already exist

-- SQLite doesn't support IF NOT EXISTS for columns,
-- so we use a workaround via a separate migration check in adapter.py