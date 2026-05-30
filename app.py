"""
app.py — Flask application factory.

Registers blueprints, configures error handlers, logging, and static file routes.
"""

import logging
import os
import shutil
from logging.handlers import RotatingFileHandler

from flask import Flask, jsonify, render_template, send_from_directory

import config
from db import adapter as db


def create_app() -> Flask:
    app = Flask(__name__, template_folder='templates', static_folder='static')

    # ── Flask config ──────────────────────────────────────────────────────────
    app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
    app.config['SECRET_KEY']         = config.SECRET_KEY
    app.config['DEBUG']              = config.DEBUG

    # ── Logging ───────────────────────────────────────────────────────────────
    _configure_logging(app)

    # ── Startup health checks ─────────────────────────────────────────────────
    _health_checks(app)

    # ── Database ──────────────────────────────────────────────────────────────
    db.init_db()

    # ── Blueprints ────────────────────────────────────────────────────────────
    from routes.upload  import upload_bp
    from routes.results import results_bp
    from routes.history import history_bp

    app.register_blueprint(upload_bp)
    app.register_blueprint(results_bp)
    app.register_blueprint(history_bp)

    # ── Root route → upload screen ────────────────────────────────────────────
    @app.route('/')
    def index():
        return render_template('screens/upload.html')

    # ── Static file serving for uploads and results ───────────────────────────
    @app.route('/uploads/<path:filename>')
    def serve_upload(filename: str):
        """Serve raw uploaded files (internal use — not a public directory listing)."""
        return send_from_directory(config.UPLOAD_DIR, filename)

    @app.route('/results/<path:filename>')
    def serve_result(filename: str):
        """Serve annotated heatmap images."""
        return send_from_directory(config.RESULTS_DIR, filename)

    # ── Error handlers ────────────────────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        if _wants_json():
            return jsonify({'error': str(e.description)}), 400
        return render_template('errors/404.html',
                               title='Bad Request',
                               message=str(e.description)), 400

    @app.errorhandler(404)
    def not_found(e):
        if _wants_json():
            return jsonify({'error': 'Not found'}), 404
        return render_template('errors/404.html',
                               title='Page Not Found',
                               message='The page you requested does not exist.'), 404

    @app.errorhandler(413)
    def file_too_large(e):
        if _wants_json():
            return jsonify({
                'error': (
                    f'File size exceeds the {config.MAX_FILE_SIZE_MB} MB limit. '
                    f'Please compress or re-scan the document.'
                )
            }), 413
        return render_template('errors/404.html',
                               title='File Too Large',
                               message=f'Maximum upload size is {config.MAX_FILE_SIZE_MB} MB.'), 413

    @app.errorhandler(500)
    def server_error(e):
        app.logger.error('Unhandled 500 error: %s', e, exc_info=True)
        if _wants_json():
            return jsonify({'error': 'An unexpected error occurred. Please try again.'}), 500
        return render_template('errors/500.html'), 500

    return app


# ── Logging configuration ─────────────────────────────────────────────────────

def _configure_logging(app: Flask) -> None:
    """Set up rotating file handler + console handler."""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Rotating file handler
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if config.DEBUG else logging.INFO)

    # Apply to root logger so all modules inherit it
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    app.logger.info("Logging initialised — log file: %s", config.LOG_FILE)


# ── Startup health checks ─────────────────────────────────────────────────────

def _health_checks(app: Flask) -> None:
    """Verify critical system dependencies exist. Fail fast with clear logs."""
    ok = True

    if not shutil.which('tesseract'):
        app.logger.warning(
            "HEALTH CHECK — Tesseract OCR binary not found. "
            "OCR will be disabled. Install: sudo apt install tesseract-ocr"
        )
        # Not fatal — OCR is a supporting signal

    poppler_ok = shutil.which('pdftoppm') or shutil.which('pdfinfo')
    if not poppler_ok:
        app.logger.warning(
            "HEALTH CHECK — poppler-utils not found. "
            "PDF uploads will be rejected. Install: sudo apt install poppler-utils"
        )

    for directory in (config.UPLOAD_DIR, config.RESULTS_DIR, config.MODELS_DIR):
        if not os.path.isdir(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                app.logger.error("HEALTH CHECK — cannot create directory %s: %s", directory, exc)
                ok = False

    if ok:
        app.logger.info("HEALTH CHECK — all directories OK")


# ── JSON detection helper ─────────────────────────────────────────────────────

def _wants_json() -> bool:
    """Return True if the request prefers a JSON response."""
    from flask import request
    return (
        request.path.startswith('/api/')
        or request.accept_mimetypes.best == 'application/json'
    )