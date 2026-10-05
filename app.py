from flask import Flask, render_template, request, send_from_directory, session, send_file, redirect, url_for, flash
import os
import re
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.utils import secure_filename

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from database.init_db import init_db
from utils.helper_functions import save_detection, get_statistics, get_all_detections, get_detection_by_id
from utils.auth import register_user, verify_login, get_user_by_id

# Import predict_image (loads CNN model)
from models.predict import predict_image

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "pharmaguard-secret-key-change-in-production")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Rate limiting: max 10 uploads per minute per user
_upload_times = {}

def _check_rate_limit(user_id):
    now = datetime.now()
    times = _upload_times.get(user_id, [])
    times = [t for t in times if (now - t).seconds < 60]
    if len(times) >= 10:
        return False
    times.append(now)
    _upload_times[user_id] = times
    return True

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
init_db()

print("PharmaGuard ready!")


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def current_user():
    uid = session.get("user_id")
    return get_user_by_id(uid) if uid else None


app.jinja_env.globals["current_user"] = current_user

def jinja2_match_test(s, pattern):
    """Custom Jinja2 test for regex matching."""
    if s is None:
        return False
    return bool(re.search(pattern, str(s)))

app.jinja_env.tests['match'] = jinja2_match_test


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Public routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name    = request.form.get("name", "").strip()
        email   = request.form.get("email", "").strip()
        subject = request.form.get("subject", "").strip()
        message = request.form.get("message", "").strip()
        if not name or not email or not message:
            flash("Please fill in all required fields.", "danger")
        else:
            # Message stored — extend here to save to DB or send email if needed
            flash("Thank you! Your message has been received. We'll get back to you soon.", "success")
            return redirect(url_for("contact"))
    return render_template("contact.html")


@app.route("/diagnostics")
def camera_diagnostics():
    """Camera system diagnostic page for troubleshooting"""
    return render_template("camera_diagnostics.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "user_id" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        name  = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password  = request.form.get("password", "")
        password2 = request.form.get("password2", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
        elif password != password2:
            flash("Passwords do not match.", "danger")
        elif len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
        else:
            ok, error = register_user(name, email, password, phone or None)
            if ok:
                flash("Account created successfully! Please log in.", "success")
                return redirect(url_for("login"))
            else:
                flash(error, "danger")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("index"))
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = verify_login(email, password)
        if user:
            session["user_id"]    = user["id"]
            session["user_name"]  = user["name"]
            session["user_phone"] = user.get("phone", "")
            flash(f"Welcome back, {user['name']}!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index"))
        else:
            flash("Invalid email or password.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


# ── Protected routes ──────────────────────────────────────────────────────────

@app.route("/detection")
@login_required
def detection():
    return render_template("detection.html")


@app.route("/detect", methods=["POST"])
@login_required
def detect():
    try:
        if not _check_rate_limit(session.get("user_id")):
            return render_template("detection.html", error="Too many uploads. Please wait a moment before scanning again.")

        if "image" not in request.files:
            return render_template("detection.html", error="No file uploaded. Please select an image.")

        file = request.files["image"]
        if not file.filename:
            return render_template("detection.html", error="No file selected.")
        if not allowed_file(file.filename):
            return render_template("detection.html", error="Invalid file type. Please upload PNG, JPG, JPEG, WEBP or BMP.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        detection_result = predict_image(filepath)
        # Use the multi-layered verdict for the prediction status
        pred_status = detection_result.get("verdict", "Unknown")
        raw_conf = detection_result.get("confidence")
        conf = round(raw_conf, 2) if raw_conf is not None else None

        result = {
            "prediction": pred_status,
            "confidence": conf,
            "medicine_name": detection_result["medicine_name"],
            "expiry_date": detection_result.get("expiry_date"),
            "expiry_status": detection_result.get("expiry_status"),
            "barcode": None,
            "text": detection_result.get("extracted_text", "")
        }
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        uid      = session.get("user_id")

        det_id = save_detection(
            date=date_str,
            image=filename,
            result=result["prediction"],
            confidence=result.get("confidence"),
            medicine_name=result.get("medicine_name"),
            expiry_date=result.get("expiry_date"),
            expiry_status=result.get("expiry_status"),
            barcode=result.get("barcode"),
            extracted_text=result.get("text"),
            user_id=uid,
            batch_no=detection_result.get("batch_number"),
            ocr_confidence=detection_result.get("ocr_confidence")
        )

        session["last_detection_id"] = det_id

        return render_template(
            "result.html",
            result=result["prediction"],
            text=result.get("text", ""),
            image=filename,
            confidence=result.get("confidence"),
            barcode=result.get("barcode"),
            expiry_date=result.get("expiry_date"),
            expiry_status=result.get("expiry_status"),
            medicine_name=result.get("medicine_name"),
            batch_number=detection_result.get("batch_number"),
            ocr_confidence=detection_result.get("ocr_confidence")
        )
    except Exception as e:
        return render_template("detection.html", error=f"An error occurred: {str(e)}")


@app.route("/dashboard")
@login_required
def dashboard():
    import sqlite3
    from config import DATABASE

    uid = session.get("user_id")
    total_scans, fake_count, real_count = get_statistics(user_id=uid)
    detections = get_all_detections(user_id=uid)

    # Get statistics by time period
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    now = datetime.now()
    week_start = now - timedelta(days=7)
    month_start = now - timedelta(days=30)

    # Week stats
    cursor.execute("""
        SELECT COUNT(*), SUM(CASE WHEN result='Real' THEN 1 ELSE 0 END),
               SUM(CASE WHEN result='Fake' THEN 1 ELSE 0 END)
        FROM detections
        WHERE user_id=? AND date >= ?
    """, (uid, week_start.strftime("%Y-%m-%d")))
    week_data = cursor.fetchone()
    week_total, week_real, week_fake = week_data[0] or 0, week_data[1] or 0, week_data[2] or 0

    # Month stats
    cursor.execute("""
        SELECT COUNT(*), SUM(CASE WHEN result='Real' THEN 1 ELSE 0 END),
               SUM(CASE WHEN result='Fake' THEN 1 ELSE 0 END)
        FROM detections
        WHERE user_id=? AND date >= ?
    """, (uid, month_start.strftime("%Y-%m-%d")))
    month_data = cursor.fetchone()
    month_total, month_real, month_fake = month_data[0] or 0, month_data[1] or 0, month_data[2] or 0

    # Year stats
    year_start = now - timedelta(days=365)
    cursor.execute("""
        SELECT COUNT(*), SUM(CASE WHEN result='Real' THEN 1 ELSE 0 END),
               SUM(CASE WHEN result='Fake' THEN 1 ELSE 0 END)
        FROM detections
        WHERE user_id=? AND date >= ?
    """, (uid, year_start.strftime("%Y-%m-%d")))
    year_data = cursor.fetchone()
    year_total, year_real, year_fake = year_data[0] or 0, year_data[1] or 0, year_data[2] or 0

    conn.close()

    return render_template("dashboard.html",
                           real_count=real_count,
                           fake_count=fake_count,
                           total_scans=total_scans,
                           detections=detections,
                           week_total=week_total,
                           week_real=week_real,
                           week_fake=week_fake,
                           month_total=month_total,
                           month_real=month_real,
                           month_fake=month_fake,
                           year_total=year_total,
                           year_real=year_real,
                           year_fake=year_fake)


@app.route("/report")
@login_required
def report():
    det_id = session.get("last_detection_id")
    if not det_id:
        # Fallback to older session key if it exists (for transition)
        old_data = session.get("last_detection")
        if old_data:
            info = {
                "medicine_name": old_data.get("medicine_name") or "Unable to read",
                "expiry_date":   old_data.get("expiry_date")   or "Unable to read",
            }
            return render_template("report.html", info=info, result=old_data["result"], 
                                 confidence=old_data.get("confidence", "N/A"), text=old_data.get("text", ""))
        return render_template("detection.html", error="No recent detection found. Please scan a medicine first.")
    
    det = get_detection_by_id(det_id)
    if not det:
        return render_template("detection.html", error="Detection record not found.")

    info = {
        "medicine_name": det.get("medicine_name") or "Unable to read",
        "expiry_date":   det.get("expiry_date")   or "Unable to read",
    }
    return render_template(
        "report.html",
        info=info,
        result=det["result"],
        confidence=det.get("confidence", "N/A"),
        text=det.get("text", "")
    )


@app.route("/download_report")
@login_required
def download_report():
    from fpdf import FPDF
    import io

    det_id = session.get("last_detection_id")
    det = get_detection_by_id(det_id) if det_id else None
    
    # Fallback to session data if DB fails or old session
    if not det:
        det = session.get("last_detection")
        
    if not det:
        return "No report available. Please perform a detection first.", 404

    pdf = FPDF()
    def safe(text):
        if not text:
            return ""
        return str(text).encode("latin-1", errors="replace").decode("latin-1")

    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Medicine Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 8, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    user = current_user()
    if user:
        pdf.cell(200, 8, txt=safe(f"Reported by: {user['name']} ({user['email']})"), ln=True, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(200, 10, txt="Medicine Details", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 9, txt=safe(f"Medicine Name : {det.get('medicine_name') or 'Unable to read'}"), ln=True)
    pdf.cell(200, 9, txt=safe(f"Expiry Date   : {det.get('expiry_date')   or 'Unable to read'}"), ln=True)
    if det.get("barcode"):
        pdf.cell(200, 9, txt=safe(f"Barcode       : {det['barcode']}"), ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "B", 13)
    pdf.cell(200, 10, txt="Analysis Result", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 9, txt=safe(f"Detection Result : {det['result']}"), ln=True)
    confidence = det.get("confidence")
    pdf.cell(200, 9, txt=safe(f"Confidence       : {f'{confidence}%' if confidence is not None else 'N/A'}"), ln=True)
    if det.get("date"):
        pdf.cell(200, 9, txt=safe(f"Scan Date        : {det['date']}"), ln=True)
    pdf.ln(5)

    text = det.get("text", "")
    if text:
        pdf.set_font("Arial", "B", 13)
        pdf.cell(200, 10, txt="Extracted Text", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 8, txt=safe(text))

    pdf_output = io.BytesIO(pdf.output(dest='S').encode('latin-1'))
    pdf_output.seek(0)

    return send_file(
        pdf_output,
        as_attachment=True,
        download_name="medicine_report.pdf",
        mimetype="application/pdf"
    )


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)