"""
pipeline/detector.py — Model loading, inference, and Grad-CAM heatmap generation.
Uses 3 fallback strategies for heatmap generation to ensure compatibility
with tensorflow-macos on Apple Silicon.
"""

import logging
import os
import tempfile

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_model        = None
_model_loaded = False


def _load_model():
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
        logger.info("Loading model from %s ...", config.MODEL_PATH)

        try:
            _model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
            logger.info("Model loaded successfully.")
        except Exception as e1:
            keras_path = config.MODEL_PATH.replace('.h5', '.keras')
            if os.path.exists(keras_path):
                _model = tf.keras.models.load_model(keras_path, compile=False)
                logger.info("Model loaded from .keras format.")
            else:
                raise e1

        logger.info(
            "Model ready: input=%s output=%s",
            _model.input_shape, _model.output_shape
        )

    except Exception as exc:
        logger.error("Failed to load model: %s — MOCK mode.", exc)
        _model = None

    _model_loaded = True
    return _model


_load_model()


# ── Public interface ──────────────────────────────────────────────────────────

def detect(preprocessed: dict, document_id: str) -> dict:
    model     = _load_model()
    ela_score = _compute_ela_score(preprocessed.get('ela_image'))

    if model is None:
        return _mock_result(ela_score)

    try:
        cnn_confidence = _run_inference(model, preprocessed['image_cv'])
        logger.info(
            "Inference: document_id=%s cnn=%.4f ela=%.4f",
            document_id, cnn_confidence, ela_score
        )
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        return _ela_only_result(ela_score)

    # Try heatmap generation with multiple fallback strategies
    heatmap_path = _try_generate_heatmap(
        preprocessed['image_cv'],
        preprocessed.get('ela_image'),
        model,
        document_id
    )

    detections = _build_detections(cnn_confidence, ela_score)
    verdict    = 'forged' if cnn_confidence >= config.VERDICT_THRESHOLD else 'authentic'

    logger.info(
        "Detection done: document_id=%s verdict=%s confidence=%.3f heatmap=%s",
        document_id, verdict, cnn_confidence,
        'yes' if heatmap_path else 'no'
    )

    return {
        'verdict':      verdict,
        'confidence':   cnn_confidence,
        'heatmap_path': heatmap_path,
        'detections':   detections,
        'ela_score':    ela_score,
    }


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(model, image_cv: np.ndarray) -> float:
    img   = image_cv.astype(np.float32) / 255.0
    img   = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)
    logger.debug("Raw predictions: %s", preds)

    if preds.shape[-1] == 2:
        confidence = float(preds[0][1])
    elif preds.shape[-1] == 1:
        confidence = float(preds[0][0])
    else:
        confidence = float(np.max(preds[0][1:]))

    return round(min(max(confidence, 0.0), 1.0), 4)


# ── Heatmap generation (3 strategies) ────────────────────────────────────────

def _try_generate_heatmap(image_cv, ela_image, model, document_id) -> str | None:
    """Try 3 strategies in order. Return path on first success."""

    # Strategy 1 — Grad-CAM with GradientTape
    try:
        path = _gradcam_strategy(image_cv, model, document_id)
        if path:
            logger.info("Heatmap: Strategy 1 (Grad-CAM) succeeded")
            return path
    except Exception as e:
        logger.warning("Heatmap Strategy 1 failed: %s", e)

    # Strategy 2 — Activation map (no gradients)
    try:
        path = _activation_map_strategy(image_cv, model, document_id)
        if path:
            logger.info("Heatmap: Strategy 2 (activation map) succeeded")
            return path
    except Exception as e:
        logger.warning("Heatmap Strategy 2 failed: %s", e)

    # Strategy 3 — ELA-based heatmap (always works)
    try:
        path = _ela_heatmap_strategy(image_cv, ela_image, document_id)
        if path:
            logger.info("Heatmap: Strategy 3 (ELA overlay) succeeded")
            return path
    except Exception as e:
        logger.warning("Heatmap Strategy 3 failed: %s", e)

    logger.warning("All heatmap strategies failed for document_id=%s", document_id)
    return None


def _gradcam_strategy(image_cv: np.ndarray, model, document_id: str) -> str | None:
    """Standard Grad-CAM using GradientTape."""
    import tensorflow as tf

    # Find last conv layer
    last_conv = _find_last_conv(model)
    if last_conv is None:
        raise ValueError("No conv layer found")

    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [last_conv.output, model.output]
    )

    img_tensor = tf.constant(
        np.expand_dims(image_cv.astype(np.float32) / 255.0, axis=0),
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        tape.watch(conv_out)
        loss = preds[:, 1] if preds.shape[-1] >= 2 else preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise ValueError("Gradients are None")

    pooled    = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_vals = conv_out.numpy()[0].copy()

    for i, w in enumerate(pooled):
        conv_vals[:, :, i] *= w

    heatmap = np.mean(conv_vals, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return _overlay_and_save(heatmap, image_cv, document_id)


def _activation_map_strategy(image_cv: np.ndarray, model, document_id: str) -> str | None:
    """Use raw activation map of last conv layer — no gradients needed."""
    import tensorflow as tf

    last_conv = _find_last_conv(model)
    if last_conv is None:
        raise ValueError("No conv layer found")

    activation_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = last_conv.output
    )

    img_batch   = np.expand_dims(image_cv.astype(np.float32) / 255.0, axis=0)
    activations = activation_model.predict(img_batch, verbose=0)

    heatmap = np.mean(activations[0], axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return _overlay_and_save(heatmap, image_cv, document_id)


def _ela_heatmap_strategy(image_cv: np.ndarray, ela_image, document_id: str) -> str | None:
    """Generate heatmap directly from ELA image — always works."""
    if ela_image is not None and ela_image.size > 0:
        # Use existing ELA image
        if ela_image.ndim == 3:
            ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
        else:
            ela_gray = ela_image
        heatmap = ela_gray.astype(np.float32) / 255.0
    else:
        # Generate ELA from scratch
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
        ela_fd, ela_path = tempfile.mkstemp(suffix='_ela.jpg')
        os.close(tmp_fd)
        os.close(ela_fd)
        try:
            cv2.imwrite(tmp_path, image_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(ela_path, image_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
            orig       = cv2.imread(tmp_path)
            comp       = cv2.imread(ela_path)
            diff       = cv2.absdiff(orig, comp)
            ela_amp    = np.clip(diff.astype(np.float32) * 20, 0, 255)
            ela_gray   = cv2.cvtColor(ela_amp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            heatmap    = ela_gray.astype(np.float32) / 255.0
        finally:
            for p in [tmp_path, ela_path]:
                try: os.remove(p)
                except OSError: pass

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return _overlay_and_save(heatmap, image_cv, document_id)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_last_conv(model):
    """Find the last Conv2D layer in a model."""
    import tensorflow as tf
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        if 'conv' in layer.__class__.__name__.lower() and hasattr(layer, 'filters'):
            return layer
    return None


def _overlay_and_save(heatmap: np.ndarray, image_cv: np.ndarray, document_id: str) -> str:
    """Resize heatmap, apply JET colormap, blend with original, and save."""
    h, w = image_cv.shape[:2]

    heatmap_uint8   = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    coloured        = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay         = cv2.addWeighted(image_cv, 0.6, coloured, 0.4, 0)

    filename    = f"{document_id}_annotated.jpg"
    output_path = os.path.join(config.RESULTS_DIR, filename)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    cv2.imwrite(output_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return output_path


def _compute_ela_score(ela_image: np.ndarray) -> float:
    if ela_image is None or ela_image.size == 0:
        return 0.0
    mean_val = float(np.mean(ela_image)) / 255.0
    return round(min(mean_val * config.ELA_AMPLIFY, 1.0), 4)


def _build_detections(cnn_confidence: float, ela_score: float) -> list:
    detections = []
    if ela_score > 0.3:
        detections.append({
            'type':        'ela_artifact',
            'confidence':  round(ela_score, 2),
            'region':      None,
            'description': (
                f"Elevated Error Level Analysis residual detected "
                f"(score={ela_score:.2f}). May indicate double-compression "
                f"or image splicing."
            ),
        })
    if cnn_confidence >= config.VERDICT_THRESHOLD:
        detections.append({
            'type':        'pattern_anomaly',
            'confidence':  round(cnn_confidence, 2),
            'region':      None,
            'description': (
                f"CNN model detected visual inconsistencies characteristic "
                f"of forged documents (confidence={cnn_confidence:.0%})."
            ),
        })
    return detections


def _mock_result(ela_score: float) -> dict:
    return {
        'verdict':      'unknown',
        'confidence':   0.0,
        'heatmap_path': None,
        'detections':   [],
        'ela_score':    ela_score,
    }


def _ela_only_result(ela_score: float) -> dict:
    verdict = 'forged' if ela_score >= config.VERDICT_THRESHOLD else 'authentic'
    detections = []
    if ela_score > 0.3:
        detections.append({
            'type':        'ela_artifact',
            'confidence':  round(ela_score, 2),
            'region':      None,
            'description': f"ELA score={ela_score:.2f} — CNN unavailable.",
        })
    return {
        'verdict':      verdict,
        'confidence':   ela_score,
        'heatmap_path': None,
        'detections':   detections,
        'ela_score':    ela_score,
    }