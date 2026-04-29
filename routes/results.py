"""
routes/results.py — Result retrieval endpoints.

Routes:
    GET /api/result/<id>   — return full result as JSON
    GET /result/<id>       — serve results screen (S-03 / S-05) rendered by Jinja2
"""

import logging

from flask import Blueprint, jsonify, render_template

from db import adapter as db

logger     = logging.getLogger(__name__)
results_bp = Blueprint('results', __name__)


# ── GET /api/result/<id> ──────────────────────────────────────────────────────

@results_bp.route('/api/result/<doc_id>', methods=['GET'])
def api_get_result(doc_id: str):
    """
    Return the full stored result for a given document UUID.
    Includes uploaded_at and filename_orig in addition to analysis data.

    Responses:
        200 — full result JSON
        404 — {"error": "Not found"}
    """
    result = db.get_result(doc_id)
    if result is None:
        return jsonify({'error': 'Result not found. It may have been deleted.'}), 404

    # Build annotated image URL (relative URL, safe for <img src="">)
    annotated_image = result.get('annotated_image')
    annotated_url   = f"/results/{annotated_image}" if annotated_image else None

    response = {
        'status':               result.get('status'),
        'result_id':            result.get('document_id'),
        'verdict':              result.get('verdict'),
        'confidence':           result.get('confidence'),
        'detections':           result.get('detections', []),
        'annotated_image_url':  annotated_url,
        'ocr_text':             result.get('ocr_text', ''),
        'processing_ms':        result.get('processing_ms'),
        'filename_orig':        result.get('filename_orig'),
        'uploaded_at':          result.get('uploaded_at'),
        'analysed_at':          result.get('analysed_at'),
        'model_version':        result.get('model_version'),
    }

    logger.debug("Result fetched: document_id=%s verdict=%s", doc_id, result.get('verdict'))
    return jsonify(response), 200


# ── GET /result/<id> ──────────────────────────────────────────────────────────

@results_bp.route('/result/<doc_id>', methods=['GET'])
def result_screen(doc_id: str):
    """
    Serve the results HTML page (S-03 or S-05 depending on navigation context).
    Jinja2 renders the page with data from the DB embedded directly.
    """
    result = db.get_result(doc_id)
    if result is None:
        return render_template(
            'errors/404.html',
            title='Result Not Found',
            message='This result no longer exists. It may have been deleted.',
        ), 404

    annotated_image  = result.get('annotated_image')
    annotated_url    = f"/results/{annotated_image}" if annotated_image else None
    original_url     = f"/uploads/{result.get('filename_stored')}"

    # Convert decimal confidence (0.0–1.0) to integer percentage for display
    confidence_pct   = int(round((result.get('confidence') or 0.0) * 100))

    return render_template(
        'screens/results.html',
        doc_id          = doc_id,
        verdict         = result.get('verdict', 'unknown'),
        confidence_pct  = confidence_pct,
        detections      = result.get('detections', []),
        annotated_url   = annotated_url,
        original_url    = original_url,
        filename_orig   = result.get('filename_orig', ''),
        uploaded_at     = result.get('uploaded_at', ''),
        analysed_at     = result.get('analysed_at', ''),
        processing_ms   = result.get('processing_ms'),
        from_history    = False,   # S-03 (live result, not S-05 saved view)
    )