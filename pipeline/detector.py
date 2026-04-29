"""
pipeline/detector.py — Model loading, inference, and Grad-CAM heatmap generation.

Contract:
    Input:  preprocessed dict from preprocessor.py
    Output: dict {
        'verdict':      'authentic' | 'forged' | 'unknown',
        'confidence':   float,          # 0.0 – 1.0
        'heatmap_path': str | None,     # abs path to annotated JPEG in results/
        'detections':   list[dict],     # [{type, confidence, region, description}]
        'ela_score':    float,          # raw ELA signal 0.0–1.0
    }

Model is loaded ONCE at module level on first import (startup).
If model weights are absent, a mock result is returned with verdict='unknown'.
"""

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

# ── Module-level model cache ──────────────────────────────────────────────────
_model        = None
_model_loaded = False


def _load_model():
    """Load the trained Keras model once. Returns model or None if absent."""
    global _model, _model_loaded
    if _model_loaded:
        return _model

    if not os.path.exists(config.MODEL_PATH):
        logger.warning(
            "Model weights not found at %s — running in MOCK mode.",
            config.MODEL_PATH
        )
        _model_loaded = True
        return None

    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(config.MODEL_PATH)
        logger.info("Model loaded from %s", config.MODEL_PATH)
    except Exception as exc:
        logger.error("Failed to load model: %s — running in MOCK mode.", exc)
        _model = None

    _model_loaded = True
    return _model


# Trigger model load at import time (startup)
_load_model()


# ── Public interface ──────────────────────────────────────────────────────────

def detect(preprocessed: dict, document_id: str) -> dict:
    """
    Run forgery detection on a preprocessed image dict.

    Args:
        preprocessed:  output dict from preprocessor.preprocess()
        document_id:   UUID — used to name the output heatmap file

    Returns detection result dict (see module docstring).
    """
    model = _load_model()

    # ── ELA score (heuristic) ─────────────────────────────────────────────────
    ela_score = _compute_ela_score(preprocessed['ela_image'])

    # ── CNN inference ─────────────────────────────────────────────────────────
    if model is None:
        return _mock_result(ela_score)

    try:
        cnn_confidence, last_conv_output, grads = _run_inference(
            model, preprocessed['image_cv']
        )
    except Exception as exc:
        logger.error("CNN inference failed: %s — falling back to mock result.", exc)
        return _mock_result(ela_score)

    # ── Grad-CAM heatmap ──────────────────────────────────────────────────────
    heatmap_path = None
    try:
        heatmap_path = _generate_gradcam(
            preprocessed['image_cv'],
            last_conv_output,
            grads,
            document_id,
        )
    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc)

    # ── Build detections list ─────────────────────────────────────────────────
    detections = _build_detections(cnn_confidence, ela_score)

    verdict = 'forged' if cnn_confidence >= config.VERDICT_THRESHOLD else 'authentic'

    logger.info(
        "Detection complete: document_id=%s verdict=%s cnn=%.3f ela=%.3f",
        document_id, verdict, cnn_confidence, ela_score
    )

    return {
        'verdict':      verdict,
        'confidence':   cnn_confidence,
        'heatmap_path': heatmap_path,
        'detections':   detections,
        'ela_score':    ela_score,
    }


# ── CNN inference ─────────────────────────────────────────────────────────────

def _run_inference(model, image_cv: np.ndarray):
    """
    Run model inference and return (confidence, last_conv_output, grads)
    for Grad-CAM computation.
    """
    import tensorflow as tf

    # Normalise + add batch dimension
    img_array = image_cv.astype(np.float32) / 255.0
    img_batch  = np.expand_dims(img_array, axis=0)

    # Find the last convolutional layer for Grad-CAM
    last_conv_layer = _find_last_conv_layer(model)

    # Build a sub-model that outputs both the last conv layer and final prediction
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs            = tf.cast(img_batch, tf.float32)
        conv_output, preds = grad_model(inputs)
        # Assume binary classification: index 1 = forged
        pred_index        = tf.argmax(preds[0])
        class_channel     = preds[:, pred_index]

    grads  = tape.gradient(class_channel, conv_output)
    confidence = float(preds[0][1]) if preds.shape[-1] > 1 else float(preds[0][0])

    return confidence, conv_output.numpy()[0], grads.numpy()[0]


def _find_last_conv_layer(model):
    """Return the last Conv2D layer in the model."""
    import tensorflow as tf
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in model — Grad-CAM requires a CNN.")


# ── Grad-CAM heatmap ──────────────────────────────────────────────────────────

def _generate_gradcam(
    image_cv: np.ndarray,
    conv_output: np.ndarray,
    grads: np.ndarray,
    document_id: str,
) -> str | None:
    """
    Generate a Grad-CAM saliency heatmap and overlay it on the original image.
    Saves the result to results/<document_id>_annotated.jpg.

    Returns the absolute path to the saved file.
    """
    # Pool gradients over spatial dimensions
    pooled_grads = np.mean(grads, axis=(0, 1))

    # Weight feature maps by pooled gradients
    for i, w in enumerate(pooled_grads):
        conv_output[:, :, i] *= w

    heatmap = np.mean(conv_output, axis=-1)
    heatmap  = np.maximum(heatmap, 0)               # ReLU
    if heatmap.max() > 0:
        heatmap /= heatmap.max()                    # normalise 0–1

    # Resize heatmap to original image size
    h, w = image_cv.shape[:2]
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))

    # Apply colour map and overlay
    coloured = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(image_cv, 0.6, coloured, 0.4, 0)

    # Save to results directory
    filename    = f"{document_id}_annotated.jpg"
    output_path = os.path.join(config.RESULTS_DIR, filename)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    cv2.imwrite(output_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.info("Grad-CAM saved: %s", output_path)
    return output_path


# ── ELA heuristic ─────────────────────────────────────────────────────────────

def _compute_ela_score(ela_image: np.ndarray) -> float:
    """
    Convert the ELA image into a scalar score in [0, 1].
    Higher = more error level variance = more likely manipulated.
    """
    if ela_image is None or ela_image.size == 0:
        return 0.0
    mean_val = float(np.mean(ela_image)) / 255.0
    return round(min(mean_val * config.ELA_AMPLIFY, 1.0), 4)


# ── Detection list builder ────────────────────────────────────────────────────

def _build_detections(cnn_confidence: float, ela_score: float) -> list[dict]:
    """Build a structured detections list from model outputs."""
    detections = []

    if ela_score > 0.3:
        detections.append({
            'type':        'ela_artifact',
            'confidence':  round(ela_score, 2),
            'region':      None,
            'description': (
                f"High ELA residual detected (score={ela_score:.2f}). "
                f"Regions with elevated error levels may indicate double-compression "
                f"or splicing."
            ),
        })

    if cnn_confidence >= config.VERDICT_THRESHOLD:
        detections.append({
            'type':        'pattern_anomaly',
            'confidence':  round(cnn_confidence, 2),
            'region':      None,
            'description': (
                f"CNN model detected visual patterns inconsistent with authentic documents "
                f"(confidence={cnn_confidence:.0%})."
            ),
        })

    return detections


# ── Mock fallback ─────────────────────────────────────────────────────────────

def _mock_result(ela_score: float) -> dict:
    """
    Return a safe mock result when model weights are absent.
    Enables frontend and route testing without trained weights.
    """
    logger.info("Mock result returned (no model weights loaded).")
    return {
        'verdict':      'unknown',
        'confidence':   0.0,
        'heatmap_path': None,
        'detections':   [],
        'ela_score':    ela_score,
    }