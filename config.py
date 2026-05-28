import os
import shutil
import logging

# ── Base Directories ───────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_DIR  = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DB_DIR      = os.path.join(BASE_DIR, 'db')
LOG_DIR     = BASE_DIR

# ── Environment ────────────────────────────────────────────────────
ENV   = os.environ.get("APP_ENV", "development")  # development | production
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# ── Database ───────────────────────────────────────────────────────
DB_PATH     = os.path.join(BASE_DIR, 'docforge.db')
SCHEMA_PATH = os.path.join(DB_DIR, 'schema.sql')

# ── File Upload Limits ─────────────────────────────────────────────
MAX_CONTENT_LENGTH = 10 * 1024 * 1024   # 10 MB
MAX_FILE_SIZE_MB   = 10

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/png',
    'application/pdf',
}

# ── Image / Pipeline ───────────────────────────────────────────────
MODEL_INPUT_SIZE = (224, 224)
MIN_IMAGE_SIZE   = 100

ELA_QUALITY = 75
ELA_AMPLIFY = 10
PDF_DPI     = 200

# ── ML / Aggregator Configuration ──────────────────────────────────
MODEL_CONFIG = {
    "threshold": 0.25,
    "weights": {
        "cnn": 0.40,
        "ela": 0.50,
        "ocr": 0.10,
    }
}

# Backward compatibility (so your existing code doesn't break)
VERDICT_THRESHOLD = MODEL_CONFIG["threshold"]
WEIGHT_CNN = MODEL_CONFIG["weights"]["cnn"]
WEIGHT_ELA = MODEL_CONFIG["weights"]["ela"]
WEIGHT_OCR = MODEL_CONFIG["weights"]["ocr"]

# ── Model ──────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(MODELS_DIR, 'forgery_model.h5')
MODEL_VERSION    = '1.0.0'
ALLOW_MOCK_MODEL = True   # if False → fail when model missing

# ── OCR (Cross-platform safe) ──────────────────────────────────────
TESSERACT_CMD = shutil.which("tesseract")
POPPLER_PATH = shutil.which("pdftoppm")

# ── Analysis Timeout ───────────────────────────────────────────────
ANALYSIS_TIMEOUT_S = 15   # NOTE: must be enforced in backend, not just config

# ── Pagination ─────────────────────────────────────────────────────
HISTORY_PAGE_LIMIT     = 10
HISTORY_PAGE_LIMIT_MAX = 100

# ── Logging ────────────────────────────────────────────────────────
LOG_FILE         = os.path.join(LOG_DIR, 'docforge.log')
LOG_MAX_BYTES    = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 3

# ── Flask ──────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-prod')

# ── Ensure runtime directories exist ───────────────────────────────
for _dir in (UPLOAD_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_dir, exist_ok=True)