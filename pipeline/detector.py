"""
pipeline/detector.py — Per-page forgery detection with Grad-CAM heatmaps.

detect_pages() accepts the full list of page dicts from preprocessor.py
and returns a list of per-page detection results.
"""

import logging
import os
import tempfile

import cv2
import numpy as np

import config

logger        = logging.getLogger(__name__)
_model        = None
_model_loaded = False


def _load_model():
    global _model, _model_loaded
    if _model_loaded:
        return _model

    if not os.path.exists(config.MODEL_PATH):
        logger.warning("Model not found at %s — MOCK mode.", config.MODEL_PATH)
        _model_loaded = True
        return None

    try:
        import tensorflow as tf
        try:
            _model = tf.keras.models.load_model(config.MODEL_PATH, compile=False)
        except Exception:
            keras_path = config.MODEL_PATH.replace('.h5', '.keras')
            if os.path.exists(keras_path):
                _model = tf.keras.models.load_model(keras_path, compile=False)
            else:
                raise
        logger.info("Model loaded: input=%s output=%s",
                    _model.input_shape, _model.output_shape)
    except Exception as exc:
        logger.error("Model load failed: %s — MOCK mode.", exc)
        _model = None

    _model_loaded = True
    return _model


_load_model()


# ── Public interface ──────────────────────────────────────────────────────────

def detect_pages(pages: list, document_id: str) -> list:
    """
    Run detection on all pages.

    Args:
        pages:       list of page dicts from preprocessor.preprocess()
        document_id: UUID for file naming

    Returns:
        list of per-page result dicts:
        [{
            'page_number':    int,
            'verdict':        str,
            'confidence':     float,
            'heatmap_path':   str | None,
            'detections':     list,
            'ela_score':      float,
            'original_url':   str,   # for display in UI
        }]
    """
    model        = _load_model()
    page_results = []

    for page in pages:
        page_num = page['page_number']
        logger.info("Detecting page %d of %s", page_num, document_id)

        ela_score = _compute_ela_score(page.get('ela_image'))

        if model is None:
            result = _mock_result(page_num, ela_score, page)
        else:
            try:
                cnn_confidence = _run_inference(model, page['image_cv'])
            except Exception as exc:
                logger.error("Page %d inference failed: %s", page_num, exc)
                result = _ela_only_result(page_num, ela_score, page)
                page_results.append(result)
                continue

            # Heatmap
            heatmap_path = _try_generate_heatmap(
                page['image_cv'],
                page.get('ela_image'),
                model,
                f"{document_id}_page{page_num}"
            )

            detections = _build_detections(cnn_confidence, ela_score)
            verdict    = 'forged' if cnn_confidence >= config.VERDICT_THRESHOLD else 'authentic'

            # Original page image URL for display
            page_img   = page.get('page_image_path', '')
            orig_url   = f"/results/{os.path.basename(page_img)}" if page_img else None

            result = {
                'page_number':  page_num,
                'verdict':      verdict,
                'confidence':   round(cnn_confidence, 4),
                'heatmap_path': heatmap_path,
                'detections':   detections,
                'ela_score':    ela_score,
                'original_url': orig_url,
            }

            logger.info(
                "Page %d: verdict=%s confidence=%.3f ela=%.3f",
                page_num, verdict, cnn_confidence, ela_score
            )

        page_results.append(result)

    return page_results


# Backward compatibility — single page detect
def detect(preprocessed: dict, document_id: str) -> dict:
    """Legacy single-page detect. Wraps detect_pages()."""
    results = detect_pages([preprocessed], document_id)
    return results[0] if results else _mock_result(1, 0.0, preprocessed)


# ── Inference ─────────────────────────────────────────────────────────────────

def _run_inference(model, image_cv: np.ndarray) -> float:
    img   = image_cv.astype(np.float32) / 255.0
    img   = np.expand_dims(img, axis=0)
    preds = model.predict(img, verbose=0)

    if preds.shape[-1] == 2:
        return round(float(preds[0][1]), 4)
    elif preds.shape[-1] == 1:
        return round(float(preds[0][0]), 4)
    return round(float(np.max(preds[0][1:])), 4)


# ── Heatmap ───────────────────────────────────────────────────────────────────

def _try_generate_heatmap(image_cv, ela_image, model, name_stem) -> str | None:
    for strategy in [_gradcam_strategy, _activation_strategy, _ela_strategy]:
        try:
            path = strategy(image_cv, ela_image, model, name_stem)
            if path:
                return path
        except Exception as e:
            logger.warning("Heatmap strategy failed: %s", e)
    return None


def _gradcam_strategy(image_cv, ela_image, model, name_stem) -> str | None:
    import tensorflow as tf
    last_conv = _find_last_conv(model)
    if last_conv is None:
        raise ValueError("No conv layer")

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output]
    )
    img_t = tf.constant(
        np.expand_dims(image_cv.astype(np.float32) / 255.0, 0),
        dtype=tf.float32
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_t, training=False)
        tape.watch(conv_out)
        loss = preds[:, 1] if preds.shape[-1] >= 2 else preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise ValueError("Gradients None")

    pooled    = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_vals = conv_out.numpy()[0].copy()
    for i, w in enumerate(pooled):
        conv_vals[:, :, i] *= w

    heatmap = np.mean(conv_vals, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return _save_heatmap(heatmap, image_cv, name_stem)


def _activation_strategy(image_cv, ela_image, model, name_stem) -> str | None:
    import tensorflow as tf
    last_conv = _find_last_conv(model)
    if last_conv is None:
        raise ValueError("No conv layer")

    act_model   = tf.keras.Model(inputs=model.inputs, outputs=last_conv.output)
    img_batch   = np.expand_dims(image_cv.astype(np.float32) / 255.0, 0)
    activations = act_model.predict(img_batch, verbose=0)

    heatmap = np.mean(activations[0], axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return _save_heatmap(heatmap, image_cv, name_stem)


def _ela_strategy(image_cv, ela_image, model, name_stem) -> str | None:
    if ela_image is not None and ela_image.size > 0:
        if ela_image.ndim == 3:
            ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
        else:
            ela_gray = ela_image
        heatmap = ela_gray.astype(np.float32) / 255.0
    else:
        tmp1_fd, tmp1 = tempfile.mkstemp(suffix='.jpg')
        tmp2_fd, tmp2 = tempfile.mkstemp(suffix='_ela.jpg')
        os.close(tmp1_fd); os.close(tmp2_fd)
        try:
            cv2.imwrite(tmp1, image_cv, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(tmp2, image_cv, [cv2.IMWRITE_JPEG_QUALITY, 90])
            diff    = cv2.absdiff(cv2.imread(tmp1), cv2.imread(tmp2))
            ela_amp = np.clip(diff.astype(np.float32) * 20, 0, 255)
            heatmap = cv2.cvtColor(ela_amp.astype(np.uint8),
                                   cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        finally:
            for p in [tmp1, tmp2]:
                try: os.remove(p)
                except OSError: pass

    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return _save_heatmap(heatmap, image_cv, name_stem)


def _save_heatmap(heatmap, image_cv, name_stem) -> str:
    h, w            = image_cv.shape[:2]
    heatmap_uint8   = np.uint8(255 * heatmap)
    heatmap_resized = cv2.resize(heatmap_uint8, (w, h))
    coloured        = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    overlay         = cv2.addWeighted(image_cv, 0.6, coloured, 0.4, 0)

    filename    = f"{name_stem}_annotated.jpg"
    output_path = os.path.join(config.RESULTS_DIR, filename)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    cv2.imwrite(output_path, overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return output_path


def _find_last_conv(model):
    import tensorflow as tf
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    return None


def _compute_ela_score(ela_image) -> float:
    if ela_image is None or ela_image.size == 0:
        return 0.0
    return round(min(float(np.mean(ela_image)) / 255.0 * config.ELA_AMPLIFY, 1.0), 4)


def _build_detections(cnn_confidence: float, ela_score: float) -> list:
    detections = []
    if ela_score > 0.3:
        detections.append({
            'type':        'ela_artifact',
            'confidence':  round(ela_score, 2),
            'region':      None,
            'description': f"High ELA residual (score={ela_score:.2f}). Possible double-compression or splicing.",
        })
    if cnn_confidence >= config.VERDICT_THRESHOLD:
        detections.append({
            'type':        'pattern_anomaly',
            'confidence':  round(cnn_confidence, 2),
            'region':      None,
            'description': f"CNN detected visual inconsistencies (confidence={cnn_confidence:.0%}).",
        })
    return detections


def _mock_result(page_num: int, ela_score: float, page: dict) -> dict:
    page_img  = page.get('page_image_path', '')
    orig_url  = f"/results/{os.path.basename(page_img)}" if page_img else None
    return {
        'page_number':  page_num,
        'verdict':      'unknown',
        'confidence':   0.0,
        'heatmap_path': None,
        'detections':   [],
        'ela_score':    ela_score,
        'original_url': orig_url,
    }


def _ela_only_result(page_num: int, ela_score: float, page: dict) -> dict:
    page_img   = page.get('page_image_path', '')
    orig_url   = f"/results/{os.path.basename(page_img)}" if page_img else None
    verdict    = 'forged' if ela_score >= config.VERDICT_THRESHOLD else 'authentic'
    detections = []
    if ela_score > 0.3:
        detections.append({
            'type': 'ela_artifact', 'confidence': round(ela_score, 2),
            'region': None, 'description': f"ELA score={ela_score:.2f}.",
        })
    return {
        'page_number':  page_num,
        'verdict':      verdict,
        'confidence':   ela_score,
        'heatmap_path': None,
        'detections':   detections,
        'ela_score':    ela_score,
        'original_url': orig_url,
    }