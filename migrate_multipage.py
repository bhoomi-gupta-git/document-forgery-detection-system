"""
migrate_multipage.py — Add multi-page columns to existing DocForge database.
Run this ONCE before restarting the app after the multi-page update.

Usage:
    python migrate_multipage.py
"""

import sqlite3
import os
import sys

# ── Find config ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import config

print("=" * 50)
print("  DocForge — Multi-page Migration")
print("=" * 50)
print(f"\nDatabase: {config.DB_PATH}")

if not os.path.exists(config.DB_PATH):
    print("  Database not found — nothing to migrate.")
    print("  It will be created fresh when you run the app.")
    sys.exit(0)

conn = sqlite3.connect(config.DB_PATH)
conn.execute("PRAGMA foreign_keys = ON")

# ── Check existing columns ────────────────────────────────────────────────────
cursor     = conn.execute("PRAGMA table_info(analysis_results)")
columns    = {row[1] for row in cursor.fetchall()}
print(f"\nExisting columns: {', '.join(sorted(columns))}")

# ── Add missing columns ───────────────────────────────────────────────────────
migrations = [
    ("total_pages",   "ALTER TABLE analysis_results ADD COLUMN total_pages INTEGER DEFAULT 1"),
    ("pages_summary", "ALTER TABLE analysis_results ADD COLUMN pages_summary TEXT"),
    ("forged_pages",  "ALTER TABLE analysis_results ADD COLUMN forged_pages TEXT"),
]

added = []
for col_name, sql in migrations:
    if col_name not in columns:
        try:
            conn.execute(sql)
            conn.commit()
            added.append(col_name)
            print(f"  ✓ Added column: {col_name}")
        except sqlite3.Error as e:
            print(f"  ✗ Failed to add {col_name}: {e}")
    else:
        print(f"  — Column already exists: {col_name}")

conn.close()

print("\n" + "=" * 50)
if added:
    print(f"  Migration complete! Added: {', '.join(added)}")
else:
    print("  Nothing to migrate — all columns already exist.")
print("  You can now restart the app: python run.py")
print("=" * 50)