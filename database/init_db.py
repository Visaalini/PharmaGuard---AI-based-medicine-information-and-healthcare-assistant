import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE


def init_db():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        phone TEXT,
        created_at TEXT NOT NULL
    )
    """)

    # Migration: add phone column to existing users table
    user_cols = {row[1] for row in cursor.execute("PRAGMA table_info(users)")}
    if "phone" not in user_cols:
        cursor.execute("ALTER TABLE users ADD COLUMN phone TEXT")
        print("Migration: added column 'phone' to users")

    # Detections table (full schema)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT NOT NULL,
        image TEXT NOT NULL,
        result TEXT NOT NULL,
        confidence REAL,
        medicine_name TEXT,
        expiry_date TEXT,
        expiry_status TEXT,
        barcode TEXT,
        batch_no TEXT,
        extracted_text TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(detections)")}
    migrations = {
        "user_id":       "INTEGER",
        "confidence":    "REAL",
        "medicine_name": "TEXT",
        "expiry_date":   "TEXT",
        "expiry_status": "TEXT",
        "barcode":       "TEXT",
        "batch_no":      "TEXT",
        "extracted_text":"TEXT",
        "ocr_confidence":"REAL",
    }
    for col, col_type in migrations.items():
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE detections ADD COLUMN {col} {col_type}")
            print(f"Migration: added column '{col}' to detections")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully.")