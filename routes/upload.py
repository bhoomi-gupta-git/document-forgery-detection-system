"""
routes/upload.py — Upload + multi-page pipeline orchestration.
"""

import logging
import time
from datetime import datetime, timezone

from flask import Blueprint, jsonify, render_template, request

import config
from db import adapter as db
from pipeline import aggregator, detector, ocr, preprocessor
from utils import file_handler, validator

logger    = logging.getLogger(__name__)
upload_bp = Blueprint('upload', __name__)


@upload_bp.route('/api/upload', methods=['POST'])
def api_upload():
    file = request.files.get('file')

    # Validate
    result = validator.validate_upload(file)
    if not result.ok:
        return jsonify({'error': result.error}), 400

    meta   = validator.get_file_metadata(file)
    doc_id = file_handler.generate_uuid()
    now    = datetime.now(timezone.utc).isoformat()

    # Save file
    try:
        filename_stored, abs_path = file_handler.save_upload(
            file, doc_id, meta['file_ext']
        )
    except IOError as exc:
        return jsonify({'error': 'Could not save file. Please try again.'}), 500

    # Insert DB record
    doc_record = {
        'id':              doc_id,
        'filename_orig':   meta['filename_orig'],
        'filename_stored': filename_stored,
        'file_ext':        meta['file_ext'],
        'file_size_bytes': meta['file_size_bytes'],
        'mime_type':       meta['mime_type'],
        'uploaded_at':     now,
        'status':          'pending',
    }
    try:
        db.insert_document(doc_record)
    except Exception as exc:
        logger.error("DB insert failed: %s", exc)
        file_handler.delete_upload(filename_stored)
        return jsonify({'error': 'Database error. Please try again.'}), 500

    # Run pipeline
    start_ms = time.monotonic()

    try:
        db.update_document_status(doc_id, 'processing')

        # Stage 1 — Preprocess ALL pages
        pages = preprocessor.preprocess(abs_path)
        total_pages = len(pages)
        logger.info("Preprocessed %d page(s) for document_id=%s", total_pages, doc_id)

        # Stage 2 — OCR on first page only (performance)
        first_page_path = pages[0].get('page_image_path') or abs_path
        ocr_output = ocr.extract_text(first_page_path)

        # Stage 3 — Detect on ALL pages
        page_results = detector.detect_pages(pages, doc_id)

        # Stage 4 — Aggregate
        processing_ms = int((time.monotonic() - start_ms) * 1000)
        final_result  = aggregator.aggregate(
            page_results, ocr_output, doc_id, processing_ms
        )

        # Persist
        db.insert_result(final_result)
        db.update_document_status(doc_id, 'complete')

    except ValueError as exc:
        logger.warning("Pipeline validation error for %s: %s", doc_id, exc)
        _mark_error(doc_id, str(exc))
        return jsonify({'error': str(exc)}), 400

    except Exception as exc:
        logger.error("Pipeline error for %s: %s", doc_id, exc, exc_info=True)
        _mark_error(doc_id, str(exc))
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

    # Build response
    annotated_url = (
        f"/results/{final_result['annotated_image']}"
        if final_result.get('annotated_image') else None
    )

    # Build per-page URLs for response
    pages_data = []
    for p in final_result.get('pages_summary', []):
        pages_data.append({
            'page_number':   p['page_number'],
            'verdict':       p['verdict'],
            'confidence':    p['confidence'],
            'original_url':  f"/results/{p['original_file']}" if p.get('original_file') else None,
            'heatmap_url':   f"/results/{p['heatmap_file']}"  if p.get('heatmap_file') else None,
            'detections':    p.get('detections', []),
        })

    response = {
        'status':              'complete',
        'result_id':           doc_id,
        'verdict':             final_result['verdict'],
        'confidence':          final_result['confidence'],
        'detections':          final_result['detections'],
        'annotated_image_url': annotated_url,
        'processing_ms':       final_result['processing_ms'],
        'total_pages':         final_result.get('total_pages', 1),
        'forged_pages':        final_result.get('forged_pages', []),
        'pages':               pages_data,
    }

    logger.info(
        "Upload complete: document_id=%s verdict=%s pages=%d processing_ms=%d",
        doc_id, final_result['verdict'],
        final_result.get('total_pages', 1),
        final_result['processing_ms']
    )
    return jsonify(response), 200


@upload_bp.route('/analyse', methods=['GET'])
def analyse_screen():
    result_id = request.args.get('result_id', '')
    return render_template('screens/progress.html', result_id=result_id)


@upload_bp.route('/analyse/status', methods=['GET'])
def analyse_status():
    result_id = request.args.get('result_id', '')
    if not result_id:
        return jsonify({'error': 'result_id required'}), 400

    doc = db.get_document(result_id)
    if doc is None:
        return jsonify({'error': 'Not found'}), 404

    status    = doc['status']
    stage_map = {'pending': 0, 'processing': 2, 'complete': 3, 'error': 3}

    return jsonify({'stage': stage_map.get(status, 0), 'status': status}), 200


@upload_bp.route('/analyse/<doc_id>', methods=['DELETE'])
def cancel_analysis(doc_id: str):
    doc = db.get_document(doc_id)
    if doc is None:
        return jsonify({'error': 'Not found'}), 404
    if doc['status'] not in ('complete', 'error'):
        _mark_error(doc_id, 'Cancelled by user.')
    return jsonify({}), 200


def _mark_error(document_id: str, message: str) -> None:
    try:
        db.update_document_status(document_id, 'error')
    except Exception as exc:
        logger.error("Could not mark error for %s: %s", document_id, exc)