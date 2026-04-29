"""
pipeline/ocr.py — Tesseract OCR wrapper.

Contract:
    Input:  original image path (str) — pre-validated, on disk
    Output: dict {
        'text':   str,          # full extracted text
        'blocks': list[dict],   # each: {text, confidence, x, y, w, h}
    }

If Tesseract is not installed, returns empty result with a warning log
rather than crashing — OCR is a supporting signal, not the primary detector.
"""

import logging

import config

logger = logging.getLogger(__name__)


def extract_text(image_path: str) -> dict:
    """
    Run Tesseract on an image file and return extracted text + bounding boxes.

    Blocks with confidence < OCR_MIN_CONFIDENCE are filtered out.
    """
    try:
        import pytesseract
        from pytesseract import Output

        # Set tesseract binary path from config
        pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

    except ImportError:
        logger.warning(
            "pytesseract is not installed — OCR step will be skipped. "
            "Install with: pip install pytesseract"
        )
        return _empty_result()

    try:
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            logger.warning("OCR: could not load image at %s", image_path)
            return _empty_result()

        # Run Tesseract with full data output
        data = pytesseract.image_to_data(
            image,
            output_type=Output.DICT,
            config='--psm 3',        # automatic page segmentation
            lang='eng',
        )

        # Build block list, filtering low-confidence tokens
        blocks = []
        n_tokens = len(data['text'])
        for i in range(n_tokens):
            raw_conf = data['conf'][i]
            # pytesseract returns -1 for non-text regions
            if raw_conf < 0:
                continue
            conf_int = int(raw_conf)
            if conf_int < config.OCR_MIN_CONFIDENCE:
                continue
            text = data['text'][i].strip()
            if not text:
                continue

            blocks.append({
                'text':       text,
                'confidence': conf_int,
                'x':          data['left'][i],
                'y':          data['top'][i],
                'w':          data['width'][i],
                'h':          data['height'][i],
            })

        full_text = ' '.join(b['text'] for b in blocks)

        logger.info(
            "OCR complete: image=%s  tokens=%d  high_conf_blocks=%d",
            image_path, n_tokens, len(blocks)
        )

        return {
            'text':   full_text,
            'blocks': blocks,
        }

    except Exception as exc:
        logger.warning("OCR failed for %s: %s — using empty result", image_path, exc)
        return _empty_result()


def _empty_result() -> dict:
    """Return a safe empty OCR result when Tesseract is unavailable or fails."""
    return {'text': '', 'blocks': []}