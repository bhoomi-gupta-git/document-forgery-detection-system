"""
pipeline/aggregator.py — Aggregate per-page results into a final verdict.

Fixed: verdict now based on combined ELA+CNN score, not just CNN page verdicts.
This ensures high ELA scores (manipulation artifacts) correctly flag forgeries.
"""

import json
import logging
import os
import statistics
import uuid
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


def aggregate(
    page_results: list,
    ocr_output: dict,
    document_id: str,
    processing_ms: int,
) -> dict:
    """
    Aggregate per-page detection results into a final document verdict.
    """
    if not page_results:
        return _empty_result(document_id, processing_ms)

    total_pages = len(page_results)

    # ── Compute combined confidence per page ──────────────────────────────────
    # Use ELA + CNN weighted combination for each page
    page_combined = []
    for p in page_results:
        cnn_conf  = p.get('confidence', 0.0)
        ela_score = p.get('ela_score', 0.0)
        combined  = round(
            cnn_conf  * config.WEIGHT_CNN +
            ela_score * config.WEIGHT_ELA,
            4
        )
        page_combined.append(combined)

    # ── Overall confidence = max across pages (worst page drives verdict) ─────
    overall_confidence = round(max(page_combined), 4)

    # ── OCR adjustment ────────────────────────────────────────────────────────
    ocr_score = _compute_ocr_consistency(ocr_output)
    overall_confidence = round(
        overall_confidence * (1 - config.WEIGHT_OCR) +
        ocr_score          * config.WEIGHT_OCR,
        4
    )
    overall_confidence = min(max(overall_confidence, 0.0), 1.0)

    # ── Verdict based on combined confidence ──────────────────────────────────
    # Check if model is in mock mode (all CNN confidences are 0.0)
    all_cnn_zero = all(p.get('confidence', 0.0) == 0.0 for p in page_results)

    if all_cnn_zero:
        overall_verdict = 'unknown'
    elif overall_confidence >= config.VERDICT_THRESHOLD:
        overall_verdict = 'forged'
    else:
        overall_verdict = 'authentic'

    # ── Which pages are forged (using combined score) ─────────────────────────
    forged_page_nums = []
    for p, combined in zip(page_results, page_combined):
        page_combined_score = round(
            combined * (1 - config.WEIGHT_OCR) + ocr_score * config.WEIGHT_OCR, 4
        )
        if page_combined_score >= config.VERDICT_THRESHOLD:
            forged_page_nums.append(p['page_number'])

    # ── Build per-page summary ────────────────────────────────────────────────
    pages_summary = []
    for p, combined in zip(page_results, page_combined):
        heatmap_rel = None
        if p.get('heatmap_path'):
            heatmap_rel = os.path.basename(p['heatmap_path'])

        orig_rel = None
        if p.get('original_url'):
            orig_rel = os.path.basename(p['original_url'].lstrip('/results/'))
            # clean up double prefix if any
            if orig_rel.startswith('results/'):
                orig_rel = orig_rel[8:]

        page_verdict = 'forged' if combined >= config.VERDICT_THRESHOLD else 'authentic'
        if all_cnn_zero:
            page_verdict = 'unknown'

        pages_summary.append({
            'page_number':   p['page_number'],
            'verdict':       page_verdict,
            'confidence':    combined,
            'ela_score':     p.get('ela_score', 0.0),
            'detections':    p.get('detections', []),
            'heatmap_file':  heatmap_rel,
            'original_file': orig_rel,
        })

    # ── Best heatmap = page with highest combined confidence ──────────────────
    best_idx      = page_combined.index(max(page_combined))
    best_page     = page_results[best_idx]
    heatmap_abs   = best_page.get('heatmap_path')
    annotated_rel = os.path.basename(heatmap_abs) if heatmap_abs else None

    # ── Combined detections from all pages ────────────────────────────────────
    all_detections = []
    for p in page_results:
        for det in p.get('detections', []):
            det_copy         = dict(det)
            det_copy['page'] = p['page_number']
            all_detections.append(det_copy)

    # Add OCR detection if suspicious
    if ocr_score > 0.4 and ocr_output.get('blocks'):
        all_detections.append({
            'type':        'text_inconsistency',
            'confidence':  round(ocr_score, 2),
            'region':      None,
            'description': f"OCR text inconsistency detected (score={ocr_score:.2f}).",
            'page':        1,
        })

    result = {
        'id':              str(uuid.uuid4()),
        'document_id':     document_id,
        'verdict':         overall_verdict,
        'confidence':      overall_confidence,
        'detections':      all_detections,
        'annotated_image': annotated_rel,
        'ocr_text':        ocr_output.get('text', ''),
        'processing_ms':   processing_ms,
        'model_version':   config.MODEL_VERSION,
        'analysed_at':     datetime.now(timezone.utc).isoformat(),
        'error_message':   None,
        'total_pages':     total_pages,
        'pages_summary':   pages_summary,
        'forged_pages':    forged_page_nums,
    }

    logger.info(
        "Aggregation: document_id=%s verdict=%s confidence=%.3f "
        "pages=%d forged_pages=%s",
        document_id, overall_verdict, overall_confidence,
        total_pages, forged_page_nums
    )

    return result


def _compute_ocr_consistency(ocr_output: dict) -> float:
    blocks = ocr_output.get('blocks', [])
    if not blocks or len(blocks) < 3:
        return 0.0
    confidences = [b['confidence'] for b in blocks if 'confidence' in b]
    if len(confidences) < 3:
        return 0.0
    try:
        stdev = statistics.stdev(confidences)
        return round(min(stdev / 30.0, 1.0), 4)
    except statistics.StatisticsError:
        return 0.0


def _empty_result(document_id: str, processing_ms: int) -> dict:
    return {
        'id':              str(uuid.uuid4()),
        'document_id':     document_id,
        'verdict':         'unknown',
        'confidence':      0.0,
        'detections':      [],
        'annotated_image': None,
        'ocr_text':        '',
        'processing_ms':   processing_ms,
        'model_version':   config.MODEL_VERSION,
        'analysed_at':     datetime.now(timezone.utc).isoformat(),
        'error_message':   'No pages could be processed.',
        'total_pages':     0,
        'pages_summary':   [],
        'forged_pages':    [],
    }