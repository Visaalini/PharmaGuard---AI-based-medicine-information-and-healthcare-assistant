import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from config import DATABASE


def get_user_by_email(email):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, password, phone FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2], "password": row[3], "phone": row[4]}
    return None


def get_user_by_id(user_id):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, email, phone FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "email": row[2], "phone": row[3]}
    return None


def register_user(name, email, password, phone=None):
    """Returns (True, None) on success, (False, error_message) on failure."""
    if get_user_by_email(email):
        return False, "An account with this email already exists."
    hashed = generate_password_hash(password)
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (name, email, password, phone, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, email, hashed, phone or None, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        return False, str(e)


def verify_login(email, password):
    """Returns user dict on success, None on failure."""
    user = get_user_by_email(email)
    if user and check_password_hash(user["password"], password):
        return user
    return None
