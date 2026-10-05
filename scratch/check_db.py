import sqlite3
import os

db_path = "database.db"
if not os.path.exists(db_path):
    print(f"Database {db_path} does not exist.")
else:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables in {db_path}: {[t[0] for t in tables]}")
        
        # Check users table
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        print(f"Number of users: {user_count}")
        
        # Check detections table
        cursor.execute("SELECT COUNT(*) FROM detections")
        det_count = cursor.fetchone()[0]
        print(f"Number of detections: {det_count}")
        
        conn.close()
    except Exception as e:
        print(f"Error checking database: {e}")
