# DocForge — Intelligent Document Forgery Detection System
**MCA Major Project 2024–25 · Bhoomi Gupta**

AI-powered web application that detects forgery in uploaded document images using computer vision and machine learning.

---

## Quick Start

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install system dependencies (Ubuntu/Debian)
sudo apt install tesseract-ocr poppler-utils libmagic1

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Place trained model weights at models/forgery_model.h5
#    (without weights the app runs in mock mode — verdict=unknown)

# 5. Run the development server
python run.py
# → http://localhost:5000
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Jinja2 · Vanilla JS · CSS Custom Properties |
| Backend | Python 3.11+ · Flask 3.x |
| CV/ML | OpenCV · TensorFlow · Grad-CAM |
| OCR | Tesseract 4.x |
| Database | SQLite (WAL mode) |

## Project Structure

```
major_project/
├── app.py              Flask factory
├── run.py              Dev entry point
├── config.py           All config constants
├── routes/             Flask blueprints
├── pipeline/           CV/ML pipeline modules
├── db/                 Schema + CRUD adapter
├── utils/              Validator + file handler
├── static/             CSS + JS assets
├── templates/          Jinja2 HTML templates
└── tests/              Unit tests
```

## Success Criteria (V1)

| Metric | Target |
|--------|--------|
| Detection accuracy | ≥ 80% |
| False positive rate | ≤ 15% |
| Average analysis time | < 10s |
| History load (50+ records) | < 2s |
| System stability | 0 crashes / 20 uploads |
| Lighthouse accessibility | ≥ 90 |

## Production Deployment

```bash
# Use Gunicorn instead of Flask dev server
gunicorn -w 2 -b 0.0.0.0:5000 "run:app"

# Set environment variables
export FLASK_SECRET_KEY="<32-random-bytes>"
export FLASK_DEBUG="False"
```

> Results are indicative only and are not legally admissible as forensic evidence.