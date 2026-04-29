"""
utils/file_handler.py — Secure file save / delete / path resolution.

All files are stored under UUID names to prevent path traversal and
name collisions. Original filenames are preserved only in the database.
"""

import logging
import os
import uuid

from werkzeug.utils import secure_filename

import config

logger = logging.getLogger(__name__)


def generate_uuid() -> str:
    """Return a fresh UUID4 string (used for document_id and filename)."""
    return str(uuid.uuid4())


def save_upload(file_storage, document_id: str, file_ext: str) -> tuple[str, str]:
    """
    Save an uploaded FileStorage to disk under a UUID filename.

    Args:
        file_storage: Werkzeug FileStorage object (rewound to position 0)
        document_id:  The UUID that will become the stored filename stem
        file_ext:     Normalised extension e.g. 'jpg', 'png', 'pdf'

    Returns:
        (filename_stored, absolute_path)

    Raises:
        IOError: if the file cannot be written to disk
    """
    # Build a safe filename — UUID so there is no user input in the path
    filename_stored = f"{document_id}.{file_ext}"
    abs_path        = _upload_path(filename_stored)

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)

    try:
        file_storage.seek(0)
        file_storage.save(abs_path)
        logger.info("Saved upload: %s (%d bytes)", abs_path, os.path.getsize(abs_path))
    except Exception as exc:
        logger.error("Failed to save upload %s: %s", abs_path, exc)
        raise IOError(f"Could not save uploaded file: {exc}") from exc

    return filename_stored, abs_path


def save_annotated_image(image_array, document_id: str) -> tuple[str, str]:
    """
    Save a heatmap / annotated OpenCV image array to the results directory.

    Args:
        image_array:  numpy ndarray (BGR, uint8) from cv2
        document_id:  UUID of the parent document

    Returns:
        (filename_stored, absolute_path)
    """
    import cv2

    filename_stored = f"{document_id}_annotated.jpg"
    abs_path        = _result_path(filename_stored)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    try:
        cv2.imwrite(abs_path, image_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info("Saved annotated image: %s", abs_path)
    except Exception as exc:
        logger.error("Failed to save annotated image %s: %s", abs_path, exc)
        raise IOError(f"Could not save annotated image: {exc}") from exc

    return filename_stored, abs_path


def delete_upload(filename_stored: str) -> None:
    """Delete a raw uploaded file. Silently ignores missing files."""
    path = _upload_path(filename_stored)
    _safe_delete(path)


def delete_result(filename_stored: str) -> None:
    """Delete an annotated result image. Silently ignores missing files."""
    path = _result_path(filename_stored)
    _safe_delete(path)


def upload_abs_path(filename_stored: str) -> str:
    """Return the absolute path for a stored upload filename."""
    return _upload_path(filename_stored)


def result_abs_path(filename_stored: str) -> str:
    """Return the absolute path for a stored result filename."""
    return _result_path(filename_stored)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _upload_path(filename: str) -> str:
    safe = secure_filename(filename)
    return os.path.join(config.UPLOAD_DIR, safe)


def _result_path(filename: str) -> str:
    safe = secure_filename(filename)
    return os.path.join(config.RESULTS_DIR, safe)


def _safe_delete(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted file: %s", path)
    except Exception as exc:
        logger.warning("Could not delete file %s: %s", path, exc)