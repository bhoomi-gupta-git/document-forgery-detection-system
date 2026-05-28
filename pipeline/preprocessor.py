"""
pipeline/preprocessor.py — OpenCV preprocessing + ELA + multi-page PDF support.

Contract:
    Input:  raw file path (str)
    Output: list of page dicts, one per page:
        [{
            'page_number':        int,       # 1-indexed
            'image_cv':           np.ndarray,
            'ela_image':          np.ndarray,
            'image_path_resized': str,
            'original_width':     int,
            'original_height':    int,
            'page_image_path':    str,        # permanent path in results/
        }]

Single image (JPG/PNG) returns a list with one item (page_number=1).
"""

import logging
import os
import tempfile

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def preprocess(file_path: str) -> list:
    """
    Run preprocessing on a file. Returns a list of page dicts.
    For images: returns list with one item.
    For PDFs:   returns list with one item per page.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        page_paths = _pdf_to_images(file_path)
    else:
        page_paths = [(1, file_path)]  # (page_number, path)

    results = []
    for page_number, page_path in page_paths:
        try:
            page_result = _preprocess_single(page_path, page_number)
            results.append(page_result)
        except Exception as exc:
            logger.warning(
                "Page %d preprocessing failed: %s — skipping", page_number, exc
            )

    if not results:
        raise ValueError(
            "No pages could be processed. "
            "The file may be corrupt or all pages are too low resolution."
        )

    logger.info(
        "Preprocessing complete: file=%s total_pages=%d",
        file_path, len(results)
    )
    return results


def _preprocess_single(image_path: str, page_number: int) -> dict:
    """Preprocess a single image file and return page dict."""

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(
            f"Page {page_number} could not be decoded. "
            "File may be corrupt or unsupported format."
        )

    original_height, original_width = image.shape[:2]

    if original_width < config.MIN_IMAGE_SIZE or original_height < config.MIN_IMAGE_SIZE:
        raise ValueError(
            f"Page {page_number} resolution ({original_width}×{original_height} px) "
            f"is too low. Minimum: {config.MIN_IMAGE_SIZE}×{config.MIN_IMAGE_SIZE} px."
        )

    # Resize to model input
    target_w, target_h = config.MODEL_INPUT_SIZE
    image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # CLAHE
    lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab       = cv2.merge((l_channel, a, b))
    image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Denoise
    image_denoised = cv2.GaussianBlur(image_enhanced, (3, 3), 0)

    # Deskew
    image_deskewed = _deskew(image_denoised)

    # Save temp resized JPEG for ELA
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(tmp_fd)
    cv2.imwrite(tmp_path, image_deskewed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ELA
    ela_image = _generate_ela(tmp_path)

    logger.debug(
        "Page %d preprocessed: size=%dx%d",
        page_number, original_width, original_height
    )

    return {
        'page_number':        page_number,
        'image_cv':           image_deskewed,
        'ela_image':          ela_image,
        'image_path_resized': tmp_path,
        'original_width':     original_width,
        'original_height':    original_height,
        'page_image_path':    image_path,   # permanent path (set by _pdf_to_images)
    }


# ── ELA ───────────────────────────────────────────────────────────────────────

def _generate_ela(image_path: str) -> np.ndarray:
    ela_fd, ela_path = tempfile.mkstemp(suffix='_ela.jpg')
    os.close(ela_fd)

    original      = cv2.imread(image_path)
    cv2.imwrite(ela_path, original, [cv2.IMWRITE_JPEG_QUALITY, config.ELA_QUALITY])
    ela_compressed = cv2.imread(ela_path)

    diff          = cv2.absdiff(original, ela_compressed)
    ela_amplified = np.clip(
        diff.astype(np.float32) * config.ELA_AMPLIFY, 0, 255
    ).astype(np.uint8)

    try:
        os.remove(ela_path)
    except OSError:
        pass

    return ela_amplified


# ── Deskew ────────────────────────────────────────────────────────────────────

def _deskew(image: np.ndarray) -> np.ndarray:
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray   = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return image

    rect  = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.5 or abs(angle) > 45:
        return image

    h, w    = image.shape[:2]
    centre  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# ── PDF → Images (ALL pages) ──────────────────────────────────────────────────

def _pdf_to_images(pdf_path: str) -> list:
    """
    Convert ALL pages of a PDF to JPEGs.
    Saves permanently to results/<stem>_page<N>.jpg

    Returns: list of (page_number, abs_path) tuples
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ValueError(
            "PDF support requires pdf2image. "
            "Install: pip install pdf2image  and  sudo apt install poppler-utils"
        )

    try:
        pages = convert_from_path(pdf_path, dpi=config.PDF_DPI)
    except Exception as exc:
        raise ValueError(
            f"Could not convert PDF. May be corrupt or password-protected. ({exc})"
        ) from exc

    if not pages:
        raise ValueError("PDF appears to have no pages.")

    pdf_stem = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    page_list = []
    for i, page in enumerate(pages):
        page_number   = i + 1
        page_filename = f"{pdf_stem}_page{page_number}.jpg"
        page_path     = os.path.join(config.RESULTS_DIR, page_filename)
        page.save(page_path, 'JPEG', quality=95)
        page_list.append((page_number, page_path))
        logger.info("PDF page %d saved → %s", page_number, page_path)

    logger.info("PDF converted: %d pages total", len(page_list))
    return page_list