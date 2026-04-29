"""
pipeline/preprocessor.py — OpenCV image preprocessing + ELA generation.

Contract:
    Input:  raw file path (str)
    Output: dict {
        'image_cv':          np.ndarray,   # BGR, resized to MODEL_INPUT_SIZE
        'ela_image':         np.ndarray,   # ELA amplified difference image
        'image_path_resized': str,         # path to temp resized JPEG on disk
        'original_width':    int,
        'original_height':   int,
    }

Raises:
    ValueError  — file cannot be decoded or resolution is below threshold
    IOError     — file is unreadable
"""

import logging
import os
import tempfile

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


def preprocess(file_path: str) -> dict:
    """Run the full preprocessing chain on a document image file."""

    # ── Handle PDF: convert first page to image ───────────────────────────────
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        file_path = _pdf_to_image(file_path)

    # ── Load image ────────────────────────────────────────────────────────────
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(
            "File could not be decoded as an image. "
            "It may be corrupt or in an unsupported format."
        )

    original_height, original_width = image.shape[:2]
    logger.debug("Loaded image: %s  size=%dx%d", file_path, original_width, original_height)

    # ── Resolution guard ──────────────────────────────────────────────────────
    if original_width < config.MIN_IMAGE_SIZE or original_height < config.MIN_IMAGE_SIZE:
        raise ValueError(
            f"Image resolution ({original_width}×{original_height} px) is too low "
            f"for reliable analysis. Please upload a higher-quality scan "
            f"(minimum {config.MIN_IMAGE_SIZE}×{config.MIN_IMAGE_SIZE} px)."
        )

    # ── Resize to model input size ────────────────────────────────────────────
    target_w, target_h = config.MODEL_INPUT_SIZE
    image_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # ── CLAHE (Contrast Limited Adaptive Histogram Equalisation) ──────────────
    lab = cv2.cvtColor(image_resized, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a, b))
    image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Gaussian denoise ──────────────────────────────────────────────────────
    image_denoised = cv2.GaussianBlur(image_enhanced, (3, 3), 0)

    # ── Deskew (straighten slight rotation) ───────────────────────────────────
    image_deskewed = _deskew(image_denoised)

    # ── Save resized JPEG temporarily (needed for ELA) ────────────────────────
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
    os.close(tmp_fd)
    cv2.imwrite(tmp_path, image_deskewed, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ── Error Level Analysis ──────────────────────────────────────────────────
    ela_image = _generate_ela(tmp_path)

    logger.info(
        "Preprocessing complete: file=%s original=%dx%d",
        file_path, original_width, original_height
    )

    return {
        'image_cv':           image_deskewed,
        'ela_image':          ela_image,
        'image_path_resized': tmp_path,
        'original_width':     original_width,
        'original_height':    original_height,
    }


# ── ELA ───────────────────────────────────────────────────────────────────────

def _generate_ela(image_path: str) -> np.ndarray:
    """
    Error Level Analysis.

    Save image at JPEG quality=ELA_QUALITY, subtract from original.
    Amplify difference ×ELA_AMPLIFY. High-residual regions → potential manipulation.
    """
    # Re-save at lower quality
    ela_fd, ela_path = tempfile.mkstemp(suffix='_ela.jpg')
    os.close(ela_fd)

    original = cv2.imread(image_path)
    cv2.imwrite(ela_path, original, [cv2.IMWRITE_JPEG_QUALITY, config.ELA_QUALITY])

    # Load the re-saved version and compute difference
    ela_compressed = cv2.imread(ela_path)

    # Clamp to avoid negative values, then amplify
    diff = cv2.absdiff(original, ela_compressed)
    ela_amplified = np.clip(diff.astype(np.float32) * config.ELA_AMPLIFY, 0, 255).astype(np.uint8)

    # Clean up temp file
    try:
        os.remove(ela_path)
    except OSError:
        pass

    logger.debug("ELA generated: max_residual=%d", ela_amplified.max())
    return ela_amplified


# ── Deskew ────────────────────────────────────────────────────────────────────

def _deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct slight skew (rotation) in the image.
    Uses minimum area rectangle of the largest contour as heuristic.
    Skips correction if skew angle is < 0.5° (noise) or > 45° (non-document).
    """
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray  = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 10:
        return image                              # not enough content to deskew

    rect  = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) < 0.5 or abs(angle) > 45:
        return image                              # skip — noise or extreme tilt

    h, w = image.shape[:2]
    centre = (w // 2, h // 2)
    M      = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    logger.debug("Deskewed by %.2f°", angle)
    return rotated


# ── PDF → Image ───────────────────────────────────────────────────────────────

def _pdf_to_image(pdf_path: str) -> str:
    """
    Convert the first page of a PDF to a JPEG temp file.
    Requires poppler-utils installed on the system.

    Returns the path to the temp JPEG file.
    Raises ValueError if pdf2image / poppler is unavailable.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ValueError(
            "PDF support requires pdf2image and poppler-utils. "
            "Install with: pip install pdf2image  and  sudo apt install poppler-utils"
        )

    try:
        pages = convert_from_path(
            pdf_path,
            dpi=config.PDF_DPI,
            first_page=1,
            last_page=1,
        )
    except Exception as exc:
        raise ValueError(
            f"Could not convert PDF to image. "
            f"The file may be corrupt or password-protected. ({exc})"
        ) from exc

    if not pages:
        raise ValueError("PDF appears to have no pages.")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix='_page1.jpg')
    os.close(tmp_fd)
    pages[0].save(tmp_path, 'JPEG', quality=95)
    logger.info("PDF first-page extracted → %s", tmp_path)
    return tmp_path