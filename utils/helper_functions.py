import sqlite3
from config import DATABASE


def save_detection(date, image, result, confidence=None, medicine_name=None,
                   expiry_date=None, expiry_status=None, barcode=None, extracted_text=None,
                   user_id=None, batch_no=None, ocr_confidence=None):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO detections (user_id, date, image, result, confidence, medicine_name, expiry_date, expiry_status, barcode, extracted_text, batch_no, ocr_confidence)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, date, image, result, confidence, medicine_name, expiry_date, expiry_status, barcode, extracted_text, batch_no, ocr_confidence))
    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id


def get_statistics(user_id=None):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        if user_id:
            cursor.execute("SELECT COUNT(*) FROM detections WHERE user_id=?", (user_id,))
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM detections WHERE result='Fake' AND user_id=?", (user_id,))
            fake = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM detections WHERE result='Real' AND user_id=?", (user_id,))
            real = cursor.fetchone()[0]
        else:
            cursor.execute("SELECT COUNT(*) FROM detections")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM detections WHERE result='Fake'")
            fake = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM detections WHERE result='Real'")
            real = cursor.fetchone()[0]
        conn.close()
        return total, fake, real
    except Exception:
        return 0, 0, 0


def get_all_detections(user_id=None):
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        if user_id:
            cursor.execute(
                "SELECT date, image, result, confidence, medicine_name, expiry_date, expiry_status, batch_no, ocr_confidence FROM detections WHERE user_id=? ORDER BY id DESC LIMIT 50",
                (user_id,)
            )
        else:
            cursor.execute(
                "SELECT date, image, result, confidence, medicine_name, expiry_date, expiry_status, batch_no, ocr_confidence FROM detections ORDER BY id DESC LIMIT 50"
            )
        rows = cursor.fetchall()
        conn.close()
        return [{
            "date": r[0] or "",
            "image": r[1] or "default.png",
            "result": r[2] or "Unknown",
            "confidence": r[3] or 0.0,
            "medicine_name": r[4] or "Not Detected",
            "expiry_date": r[5] or "N/A",
            "expiry_status": r[6] or "UNKNOWN",
            "batch_no": r[7] or "N/A",
            "ocr_confidence": r[8] or 0.0
        } for r in rows]
    except Exception as e:
        print(f"Database error: {e}")
        return []


def get_detection_by_id(det_id):
    """Fetch a single detection record by ID for reports/results."""
    try:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row  # This allows accessing columns by name
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, date, image, result, confidence, medicine_name, 
                   expiry_date, expiry_status, barcode, extracted_text as text, 
                   batch_no, ocr_confidence 
            FROM detections WHERE id=?
        """, (det_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        print(f"Error fetching detection {det_id}: {e}")
        return None
