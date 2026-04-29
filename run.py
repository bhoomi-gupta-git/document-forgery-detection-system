"""
run.py — Flask development entry point.

For production, use Gunicorn instead:
    gunicorn -w 2 -b 0.0.0.0:5000 "run:app"
"""

from app import create_app

app = application = create_app()

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        use_reloader=False,   # prevents double model load on reload
    )