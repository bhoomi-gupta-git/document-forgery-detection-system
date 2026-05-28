"""
routes/results.py — Result retrieval with multi-page support.
"""

import json
import logging
import os

from flask import Blueprint, jsonify, render_template

from db import adapter as db
import config

logger     = logging.getLogger(__name__)
results_bp = Blueprint('results', __name__)


@results_bp.route('/api/result/<doc_id>', methods=['GET'])
def api_get_result(doc_id: str):
    result = db.get_result(doc_id)
    if result is None:
        return jsonify({'error': 'Result not found.'}), 404

    annotated_url = (
        f"/results/{result['annotated_image']}"
        if result.get('annotated_image') else None
    )

    # Parse pages_summary JSON if stored
    pages_summary = result.get('pages_summary') or []
    if isinstance(pages_summary, str):
        try:
            pages_summary = json.loads(pages_summary)
        except Exception:
            pages_summary = []

    return jsonify({
        'status':              result.get('status'),
        'result_id':           result.get('document_id'),
        'verdict':             result.get('verdict'),
        'confidence':          result.get('confidence'),
        'detections':          result.get('detections', []),
        'annotated_image_url': annotated_url,
        'processing_ms':       result.get('processing_ms'),
        'filename_orig':       result.get('filename_orig'),
        'uploaded_at':         result.get('uploaded_at'),
        'total_pages':         result.get('total_pages', 1),
        'forged_pages':        result.get('forged_pages', []),
        'pages_summary':       pages_summary,
    }), 200


@results_bp.route('/result/<doc_id>', methods=['GET'])
def result_screen(doc_id: str):
    result = db.get_result(doc_id)
    if result is None:
        return render_template(
            'errors/404.html',
            title='Result Not Found',
            message='This result no longer exists.'
        ), 404

    pages_summary = _parse_pages(result)
    confidence_pct = int(round((result.get('confidence') or 0.0) * 100))

    # For single-image or first page fallback
    file_ext       = result.get('file_ext', '')
    filename_stored = result.get('filename_stored', '')
    if file_ext == 'pdf':
        stem         = os.path.splitext(filename_stored)[0]
        original_url = f"/results/{stem}_page1.jpg"
    else:
        original_url = f"/uploads/{filename_stored}"

    annotated_url = (
        f"/results/{result['annotated_image']}"
        if result.get('annotated_image') else None
    )

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
        from_history    = False,
        total_pages     = result.get('total_pages', 1),
        forged_pages    = result.get('forged_pages', []),
        pages_summary   = pages_summary,
        file_ext        = file_ext,
    )


def _parse_pages(result: dict) -> list:
    """Parse pages_summary from DB result."""
    raw = result.get('pages_summary') or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []

    pages = []
    for p in raw:
        pages.append({
            'page_number':  p.get('page_number', 1),
            'verdict':      p.get('verdict', 'unknown'),
            'confidence':   p.get('confidence', 0.0),
            'confidence_pct': int(round(p.get('confidence', 0.0) * 100)),
            'detections':   p.get('detections', []),
            'original_url': f"/results/{p['original_file']}" if p.get('original_file') else None,
            'heatmap_url':  f"/results/{p['heatmap_file']}"  if p.get('heatmap_file') else None,
        })
    return pages