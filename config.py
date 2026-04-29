import os

# ── Base Directories ──────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

UPLOAD_DIR   = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR  = os.path.join(BASE_DIR, 'results')
MODELS_DIR   = os.path.join(BASE_DIR, 'models')
DB_DIR       = os.path.join(BASE_DIR, 'db')
LOG_DIR      = BASE_DIR

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH      = os.path.join(BASE_DIR, 'docforge.db')
SCHEMA_PATH  = os.path.join(DB_DIR, 'schema.sql')

# ── File Upload Limits ────────────────────────────────────────────────────────
MAX_CONTENT_LENGTH = 10 * 1024 * 1024
MAX_FILE_SIZE_MB   = 10
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
ALLOWED_MIME_TYPES = {
    'image/jpeg',
    'image/png',
    'application/pdf',
}

# ── Image / Pipeline ──────────────────────────────────────────────────────────
MODEL_INPUT_SIZE   = (224, 224)
MIN_IMAGE_SIZE     = 100
ELA_QUALITY        = 90 
ELA_AMPLIFY        = 20
PDF_DPI            = 200

# ── ML / Aggregator Thresholds ────────────────────────────────────────────────
VERDICT_THRESHOLD  = 0.5

# Weighted average weights for aggregator (must sum to 1.0)
WEIGHT_CNN         = 0.60
WEIGHT_ELA         = 0.25
WEIGHT_OCR         = 0.15

# Model weights file (place trained model here)
MODEL_PATH         = os.path.join(MODELS_DIR, 'forgery_model.h5')
MODEL_VERSION      = '1.0.0'

# ── OCR ───────────────────────────────────────────────────────────────────────
TESSERACT_CMD      = '/usr/bin/tesseract'          
OCR_MIN_CONFIDENCE = 60

# ── Analysis Timeout ──────────────────────────────────────────────────────────
ANALYSIS_TIMEOUT_S = 15                            

# ── Pagination Defaults ───────────────────────────────────────────────────────
HISTORY_PAGE_LIMIT      = 10
HISTORY_PAGE_LIMIT_MAX  = 100

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE         = os.path.join(LOG_DIR, 'docforge.log')
LOG_MAX_BYTES    = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 3

# ── Flask ─────────────────────────────────────────────────────────────────────
SECRET_KEY  = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-change-in-prod')
DEBUG       = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# ── Ensure runtime directories exist ─────────────────────────────────────────
for _dir in (UPLOAD_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_dir, exist_ok=True)