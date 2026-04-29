"""
pipeline/aggregator.py — Combine sub-detector outputs into final verdict dict.

Contract:
    Input:
        detector_output:  dict from detector.detect()
        ocr_output:       dict from ocr.extract_text()
        document_id:      str (UUID)
        processing_ms:    int (wall-clock time so far)

    Output: final result dict ready for db.adapter.insert_result()
    {
        'id':               str,   # new UUID for analysis_results row
        'document_id':      str,
        'verdict':          str,   # authentic | forged | unknown
        'confidence':       float, # 0.0–1.0
        'detections':       list,
        'annotated_image':  str | None,  # relative filename only
        'ocr_text':         str,
        'processing_ms':    int,
        'model_version':    str,
        'analysed_at':      str,  # UTC ISO-8601
        'error_message':    None,
    }
"""

import logging
import os
import uuid
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


def aggregate(
    detector_output: dict,
    ocr_output: dict,
    document_id: str,
    processing_ms: int,
) -> dict:
    """
    Merge detector + OCR signals into a final verdict using a weighted average.

    Weights (configurable in config.py):
        CNN confidence  → WEIGHT_CNN
        ELA score       → WEIGHT_ELA
        OCR consistency → WEIGHT_OCR
    """

    cnn_confidence  = detector_output.get('confidence', 0.0)
    ela_score       = detector_output.get('ela_score', 0.0)
    ocr_consistency = _compute_ocr_consistency(ocr_output)

    # ── Weighted average ──────────────────────────────────────────────────────
    if detector_output.get('verdict') == 'unknown':
        # No model loaded — fall through to ELA-only heuristic
        combined = ela_score * 0.8 + ocr_consistency * 0.2
        verdict  = 'unknown'
    else:
        combined = (
            cnn_confidence  * config.WEIGHT_CNN +
            ela_score       * config.WEIGHT_ELA +
            ocr_consistency * config.WEIGHT_OCR
        )
        verdict  = 'forged' if combined >= config.VERDICT_THRESHOLD else 'authentic'

    combined = round(min(max(combined, 0.0), 1.0), 4)

    # ── Annotated image: store relative filename, not abs path ───────────────
    heatmap_abs  = detector_output.get('heatmap_path')
    annotated_rel = os.path.basename(heatmap_abs) if heatmap_abs else None

    # ── Merge OCR text-inconsistency detection into detections list ───────────
    detections = list(detector_output.get('detections', []))
    if ocr_consistency > 0.4 and ocr_output.get('blocks'):
        detections.append({
            'type':        'text_inconsistency',
            'confidence':  round(ocr_consistency, 2),
            'region':      None,
            'description': (
                f"OCR analysis found text inconsistencies that may indicate "
                f"character-level tampering (score={ocr_consistency:.2f})."
            ),
        })

    result = {
        'id':              str(uuid.uuid4()),
        'document_id':     document_id,
        'verdict':         verdict,
        'confidence':      combined,
        'detections':      detections,
        'annotated_image': annotated_rel,
        'ocr_text':        ocr_output.get('text', ''),
        'processing_ms':   processing_ms,
        'model_version':   config.MODEL_VERSION,
        'analysed_at':     datetime.now(timezone.utc).isoformat(),
        'error_message':   None,
    }

    logger.info(
        "Aggregation complete: document_id=%s verdict=%s confidence=%.3f "
        "(cnn=%.3f ela=%.3f ocr=%.3f)",
        document_id, verdict, combined, cnn_confidence, ela_score, ocr_consistency
    )

    return result


# ── OCR consistency heuristic ─────────────────────────────────────────────────

def _compute_ocr_consistency(ocr_output: dict) -> float:
    """
    Simple heuristic: measure variance in per-token confidence scores.
    High variance → some tokens are very uncertain → possible text manipulation.

    Returns a score in [0.0, 1.0].  Higher = more suspicious.
    """
    blocks = ocr_output.get('blocks', [])
    if not blocks:
        return 0.0

    confidences = [b['confidence'] for b in blocks if 'confidence' in b]
    if len(confidences) < 3:
        return 0.0

    import statistics
    try:
        stdev = statistics.stdev(confidences)
        mean  = statistics.mean(confidences)
    except statistics.StatisticsError:
        return 0.0

    # Normalise: stdev of 30+ on a 0-100 scale is suspicious
    consistency_score = min(stdev / 30.0, 1.0)
    logger.debug(
        "OCR consistency: mean=%.1f stdev=%.1f score=%.3f",
        mean, stdev, consistency_score
    )
    return round(consistency_score, 4)