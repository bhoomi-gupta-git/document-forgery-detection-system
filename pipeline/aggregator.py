"""
pipeline/aggregator.py — Smart multi-signal verdict logic.

Uses a scoring system based on multiple signals:
- CNN confidence
- ELA score  
- OCR consistency
- Number of detections

This gives more reliable verdicts even with a 74% accuracy model.
"""

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
    if not page_results:
        return _empty_result(document_id, processing_ms)

    total_pages  = len(page_results)
    all_cnn_zero = all(p.get('confidence', 0.0) == 0.0 for p in page_results)

    # Score each page
    page_scores = [_score_page(p) for p in page_results]

    # Overall = max across pages
    overall_score = max(page_scores)

    # OCR boost
    ocr_score = _compute_ocr_consistency(ocr_output)
    if ocr_score > 0.5:
        overall_score = min(overall_score + 0.05, 1.0)

    overall_score = round(overall_score, 4)

    # Final verdict
    if all_cnn_zero:
        overall_verdict = 'unknown'
    elif overall_score >= config.VERDICT_THRESHOLD:
        overall_verdict = 'forged'
    else:
        overall_verdict = 'authentic'

    # Per-page verdicts
    forged_page_nums = []
    pages_summary    = []

    for p, score in zip(page_results, page_scores):
        if all_cnn_zero:
            page_verdict = 'unknown'
        elif score >= config.VERDICT_THRESHOLD:
            page_verdict = 'forged'
            forged_page_nums.append(p['page_number'])
        else:
            page_verdict = 'authentic'

        heatmap_rel = None
        if p.get('heatmap_path'):
            heatmap_rel = os.path.basename(p['heatmap_path'])

        orig_rel = None
        if p.get('original_url'):
            orig_rel = os.path.basename(p['original_url'])

        pages_summary.append({
            'page_number':   p['page_number'],
            'verdict':       page_verdict,
            'confidence':    round(score, 4),
            'ela_score':     p.get('ela_score', 0.0),
            'detections':    p.get('detections', []),
            'heatmap_file':  heatmap_rel,
            'original_file': orig_rel,
        })

    # Best heatmap
    best_idx      = page_scores.index(max(page_scores))
    heatmap_abs   = page_results[best_idx].get('heatmap_path')
    annotated_rel = os.path.basename(heatmap_abs) if heatmap_abs else None

    # All detections
    all_detections = []
    for p in page_results:
        for det in p.get('detections', []):
            det_copy         = dict(det)
            det_copy['page'] = p['page_number']
            all_detections.append(det_copy)

    if ocr_score > 0.4 and ocr_output.get('blocks'):
        all_detections.append({
            'type':        'text_inconsistency',
            'confidence':  round(ocr_score, 2),
            'region':      None,
            'description': f"OCR text inconsistency (score={ocr_score:.2f}).",
            'page':        1,
        })

    result = {
        'id':              str(uuid.uuid4()),
        'document_id':     document_id,
        'verdict':         overall_verdict,
        'confidence':      overall_score,
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
        "Aggregation: document_id=%s verdict=%s score=%.3f pages=%d forged=%s",
        document_id, overall_verdict, overall_score,
        total_pages, forged_page_nums
    )

    return result


def _score_page(page: dict) -> float:
    """
    Score a single page using CNN + ELA + detection count.

    Rules:
    - Base: CNN confidence * WEIGHT_CNN
    - ELA > 0.4: add ELA * WEIGHT_ELA
    - 2+ detections: +0.05 bonus
    - ELA > 0.8 AND CNN > 0.1: force minimum 0.50
    - CNN > 0.5 AND ELA > 0.5: force minimum 0.60
    """
    cnn_conf  = page.get('confidence', 0.0)
    ela_score = page.get('ela_score', 0.0)
    n_dets    = len(page.get('detections', []))

    score = cnn_conf * config.WEIGHT_CNN

    if ela_score > 0.4:
        score += ela_score * config.WEIGHT_ELA

    if n_dets >= 2:
        score += 0.05
    elif n_dets == 1:
        score += 0.02

    # Strong ELA signal overrides low CNN
    if ela_score > 0.8 and cnn_conf > 0.1:
        score = max(score, 0.50)

    # Both signals agree
    if cnn_conf > 0.5 and ela_score > 0.5:
        score = max(score, 0.60)

    return round(min(score, 1.0), 4)


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