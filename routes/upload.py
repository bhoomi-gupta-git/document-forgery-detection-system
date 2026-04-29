"""
routes/upload.py — File upload, pipeline orchestration, and progress endpoints.

Routes:
    POST   /api/upload         — validate + save + run pipeline + return result JSON
    GET    /analyse            — serve progress screen (S-02)
    GET    /analyse/status     — poll endpoint for progress.js
    DELETE /analyse/<id>       — cancel in-flight analysis (called by Cancel link)
"""

import logging
import time
import uuid
from datetime import datetime, timezone

from flask import Blueprint, jsonify, render_template, request

import config
from db import adapter as db
from pipeline import aggregator, detector, ocr, preprocessor
from utils import file_handler, validator

logger  = logging.getLogger(__name__)
upload_bp = Blueprint('upload', __name__)


# ── POST /api/upload ──────────────────────────────────────────────────────────

@upload_bp.route('/api/upload', methods=['POST'])
def api_upload():
    """
    1. Validate file
    2. Save to disk
    3. Insert pending DB record
    4. Run full pipeline
    5. Update DB record
    6. Return result JSON
    """
    file = request.files.get('file')

    # ── Validation ────────────────────────────────────────────────────────────
    result = validator.validate_upload(file)
    if not result.ok:
        logger.warning("Upload rejected: %s", result.error)
        return jsonify({'error': result.error}), 400

    meta    = validator.get_file_metadata(file)
    doc_id  = file_handler.generate_uuid()
    now_utc = datetime.now(timezone.utc).isoformat()

    # ── Save file to disk ─────────────────────────────────────────────────────
    try:
        filename_stored, abs_path = file_handler.save_upload(
            file, doc_id, meta['file_ext']
        )
    except IOError as exc:
        logger.error("File save failed: %s", exc)
        return jsonify({'error': 'Could not save uploaded file. Please try again.'}), 500

    # ── Insert pending DB record ───────────────────────────────────────────────
    doc_record = {
        'id':              doc_id,
        'filename_orig':   meta['filename_orig'],
        'filename_stored': filename_stored,
        'file_ext':        meta['file_ext'],
        'file_size_bytes': meta['file_size_bytes'],
        'mime_type':       meta['mime_type'],
        'uploaded_at':     now_utc,
        'status':          'pending',
    }
    try:
        db.insert_document(doc_record)
    except Exception as exc:
        logger.error("DB insert failed for %s: %s", doc_id, exc)
        file_handler.delete_upload(filename_stored)
        return jsonify({'error': 'Database error. Please try again.'}), 500

    # ── Run pipeline ──────────────────────────────────────────────────────────
    start_ms = time.monotonic()

    try:
        db.update_document_status(doc_id, 'processing')

        # Stage 1 — Preprocessing
        preprocessed = preprocessor.preprocess(abs_path)

        # Stage 2 — OCR
        ocr_output = ocr.extract_text(abs_path)

        # Stage 3 — Detection + heatmap
        detection_output = detector.detect(preprocessed, doc_id)

        # Stage 4 — Aggregate
        processing_ms = int((time.monotonic() - start_ms) * 1000)
        final_result  = aggregator.aggregate(
            detection_output, ocr_output, doc_id, processing_ms
        )

        # Persist result
        db.insert_result(final_result)
        db.update_document_status(doc_id, 'complete')

    except ValueError as exc:
        # User-facing validation errors from the pipeline (bad resolution, corrupt, etc.)
        logger.warning("Pipeline validation error for %s: %s", doc_id, exc)
        _mark_error(doc_id, str(exc))
        return jsonify({'error': str(exc)}), 400

    except Exception as exc:
        logger.error("Pipeline error for document_id=%s: %s", doc_id, exc, exc_info=True)
        _mark_error(doc_id, f"Unexpected analysis error: {exc}")
        return jsonify({
            'error': 'Analysis failed due to an unexpected error. Please try again.'
        }), 500

    # ── Build response ────────────────────────────────────────────────────────
    annotated_url = (
        f"/results/{final_result['annotated_image']}"
        if final_result.get('annotated_image') else None
    )

    response = {
        'status':               'complete',
        'result_id':            doc_id,
        'verdict':              final_result['verdict'],
        'confidence':           final_result['confidence'],
        'detections':           final_result['detections'],
        'annotated_image_url':  annotated_url,
        'ocr_text':             final_result.get('ocr_text', ''),
        'processing_ms':        final_result['processing_ms'],
    }

    logger.info(
        "Upload complete: document_id=%s verdict=%s confidence=%.3f processing_ms=%d",
        doc_id, final_result['verdict'], final_result['confidence'],
        final_result['processing_ms']
    )
    return jsonify(response), 200


# ── GET /analyse ──────────────────────────────────────────────────────────────

@upload_bp.route('/analyse', methods=['GET'])
def analyse_screen():
    """Serve the S-02 progress screen."""
    result_id = request.args.get('result_id', '')
    return render_template('screens/progress.html', result_id=result_id)


# ── GET /analyse/status ───────────────────────────────────────────────────────

@upload_bp.route('/analyse/status', methods=['GET'])
def analyse_status():
    """
    Poll endpoint for progress.js.
    Returns: { stage: 0-3, status: 'processing'|'complete'|'error' }
    """
    result_id = request.args.get('result_id', '')
    if not result_id:
        return jsonify({'error': 'result_id is required'}), 400

    doc = db.get_document(result_id)
    if doc is None:
        return jsonify({'error': 'Result not found'}), 404

    status = doc['status']
    # Map DB status → pipeline stage number for the stepper
    stage_map = {
        'pending':    0,
        'processing': 2,
        'complete':   3,
        'error':      3,
    }
    stage = stage_map.get(status, 0)

    return jsonify({'stage': stage, 'status': status}), 200


# ── DELETE /analyse/<id> ──────────────────────────────────────────────────────

@upload_bp.route('/analyse/<doc_id>', methods=['DELETE'])
def cancel_analysis(doc_id: str):
    """
    Cancel an in-flight analysis. Called by the Cancel link on S-02.
    Marks the DB record as error and does not delete the file (for audit trail).
    """
    doc = db.get_document(doc_id)
    if doc is None:
        return jsonify({'error': 'Not found'}), 404

    if doc['status'] in ('complete', 'error'):
        # Already terminal — nothing to cancel
        return jsonify({}), 200

    _mark_error(doc_id, 'Analysis cancelled by user.')
    logger.info("Analysis cancelled by user: document_id=%s", doc_id)
    return jsonify({}), 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mark_error(document_id: str, message: str) -> None:
    """Set document status to error. Never raises — best-effort."""
    try:
        db.update_document_status(document_id, 'error')
    except Exception as exc:
        logger.error("Could not mark document %s as error: %s", document_id, exc)