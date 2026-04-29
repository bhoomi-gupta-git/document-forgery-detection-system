"""
routes/history.py — History dashboard endpoints.

Routes:
    GET /api/history   — paginated JSON list of past analyses
    GET /history       — serve history dashboard HTML (S-04)
    GET /result/<id>/detail — serve saved result detail (S-05, from history)
"""

import logging

from flask import Blueprint, jsonify, render_template, request

import config
from db import adapter as db

logger     = logging.getLogger(__name__)
history_bp = Blueprint('history', __name__)


# ── GET /api/history ──────────────────────────────────────────────────────────

@history_bp.route('/api/history', methods=['GET'])
def api_history():
    """
    Return a paginated list of analysis records.

    Query params:
        page    (int, default 1)
        limit   (int, default 10, max 100)
        verdict (str, optional: 'authentic' | 'forged')
        sort    (str, default 'date': 'date' | 'filename' | 'verdict' | 'confidence')
        dir     (str, default 'desc': 'asc' | 'desc')
        q       (str, optional filename search)

    Response: { page, limit, total, items: [...] }
    """
    page    = _safe_int(request.args.get('page'),  1,  min_val=1)
    limit   = _safe_int(request.args.get('limit'), config.HISTORY_PAGE_LIMIT, min_val=1)
    verdict = request.args.get('verdict') or None
    sort    = request.args.get('sort', 'date')
    dir_    = request.args.get('dir',  'desc')
    search  = request.args.get('q') or None

    # Sanitise verdict — only allow known values
    if verdict not in (None, 'authentic', 'forged'):
        verdict = None

    try:
        data = db.get_history(
            page=page,
            limit=limit,
            verdict=verdict,
            sort=sort,
            direction=dir_,
            search=search,
        )
    except Exception as exc:
        logger.error("History query failed: %s", exc)
        return jsonify({'error': 'Could not load history. Please try again.'}), 500

    # Convert decimal confidence to integer percentage for each item
    for item in data.get('items', []):
        raw = item.get('confidence')
        item['confidence_pct'] = int(round((raw or 0.0) * 100))

    return jsonify(data), 200


# ── GET /history ──────────────────────────────────────────────────────────────

@history_bp.route('/history', methods=['GET'])
def history_screen():
    """Serve the S-04 history dashboard."""
    return render_template('screens/history.html')


# ── GET /result/<id>/detail ───────────────────────────────────────────────────

@history_bp.route('/result/<doc_id>/detail', methods=['GET'])
def detail_screen(doc_id: str):
    """
    Serve the S-05 saved result detail page.
    Identical layout to S-03 but with BreadcrumbBar + SavedLabel chip.
    """
    result = db.get_result(doc_id)
    if result is None:
        return render_template(
            'errors/404.html',
            title='Result Not Found',
            message='This record no longer exists. It may have been cleared.',
        ), 404

    annotated_image = result.get('annotated_image')
    annotated_url   = f"/results/{annotated_image}" if annotated_image else None
    original_url    = f"/uploads/{result.get('filename_stored')}"
    confidence_pct  = int(round((result.get('confidence') or 0.0) * 100))

    return render_template(
        'screens/detail.html',
        doc_id         = doc_id,
        verdict        = result.get('verdict', 'unknown'),
        confidence_pct = confidence_pct,
        detections     = result.get('detections', []),
        annotated_url  = annotated_url,
        original_url   = original_url,
        filename_orig  = result.get('filename_orig', ''),
        uploaded_at    = result.get('uploaded_at', ''),
        analysed_at    = result.get('analysed_at', ''),
        processing_ms  = result.get('processing_ms'),
        from_history   = True,    # S-05 — shows BreadcrumbBar + SavedLabel
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(value, default: int, min_val: int = 1) -> int:
    try:
        return max(int(value), min_val)
    except (TypeError, ValueError):
        return default