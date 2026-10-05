"""
models/predict.py  — PharmaGuard v2 Prediction Pipeline
═══════════════════════════════════════════════════════════════════════════════

Architecture
────────────
  A(I)  = CNN authenticity score (ResNet50V2-based, 0-100 %)
  T(I)  = Text-layer verification (OCR + multi-pass fuzzy matching)
  C(I)  = Company matching
  V     = A(I) + T(I) + C(I)  →  Verdict

Datasets loaded at startup (lazy, thread-safe):
  • Medicine_Details.csv         → real medicine names + manufacturers
  • medicine_dataset.csv         → extended medicine name lookup

Return dict (compatible with existing app.py):
  status, confidence, confidence_label,
  medicine_name, expiry_date, extracted_text,
  expiry_status, verdict, company_name
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import re
import logging
import threading
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("pharmaguard.predict")

# -- Paths ---------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent          # D:/pp/PHARMAGUARD
DATASET_DIR  = BASE_DIR / "dataset"

# ── CNN Model (lazy, thread-safe) ─────────────────────────────────────────────
_model      = None
_model_lock = threading.Lock()

def _load_model():
    """Load CNN model once.  Returns model or None."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        try:
            import tensorflow as tf
            from config import MODEL_CANDIDATE_PATHS
        except ImportError as exc:
            log.error("TensorFlow/config import failed: %s", exc)
            return None

        for path in MODEL_CANDIDATE_PATHS:
            try:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    _model = tf.keras.models.load_model(path)
                    log.info("CNN model loaded from: %s", path)
                    return _model
            except Exception as exc:
                log.warning("Cannot load model from %s: %s", path, exc)

        log.warning("No valid CNN model found — CNN layer disabled.")
        return None


# ── Tesseract path (adjust if installed elsewhere) ───────────────────────────
import shutil
_tess = shutil.which("tesseract")
if _tess is None:
    # Common Windows install paths
    import os
    for _p in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]:
        if os.path.exists(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            break

import pytesseract
log.info("Detector using pytesseract backend")


# -- Text normalization --------------------------------------------------------

def _normalize(text: str) -> str:
    """Uppercase, keep only alphanumeric + spaces, collapse whitespace."""
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Image preprocessing ───────────────────────────────────────────────────────

MAX_DIM = 2400   # maximum dimension before downscaling

def _remove_gloss(gray_img: np.ndarray) -> np.ndarray:
    """Remove glossy/reflective spots from medicine packaging."""
    try:
        avg = cv2.mean(gray_img)[0]
        # If already bright (product photo on white bg), skip aggressive deglossing
        if avg > 200:
            return gray_img
        _, bright_mask = cv2.threshold(gray_img, 245, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=1)
        result = cv2.inpaint(gray_img, bright_mask, 3, cv2.INPAINT_TELEA)
        return result
    except Exception:
        return gray_img

def _sharpen_image(img: np.ndarray) -> np.ndarray:
    """Apply aggressive sharpening for small/blurry text."""
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv2.filter2D(img, -1, kernel)

def _detect_text_regions(gray_img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect potential text regions for focused OCR."""
    # Adaptive threshold to find text
    thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Dilate to merge words into lines/blocks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    h_img, w_img = gray_img.shape[:2]
    # Filter for text-like regions (support both horizontal and vertical)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Horizontal lines or vertical blocks (like on bottles)
        if (50 < w < w_img * 0.9 and 20 < h < h_img * 0.3) or (20 < w < w_img * 0.3 and 50 < h < h_img * 0.9):
            regions.append((x, y, w, h))
    
    # Sort regions top-to-bottom
    regions.sort(key=lambda r: r[1])
    return regions

def _preprocess_image_variants(image_path: str) -> list[np.ndarray]:
    """
    Produce multiple enhanced grayscale variants for best OCR accuracy.
    Includes rotation correction, gloss removal, sharpening, and adaptive thresholding.
    """
    img = cv2.imread(image_path)
    if img is None:
        log.error("Cannot read image: %s", image_path)
        return []

    # 0. Downscale very large images for faster + better OCR
    h, w = img.shape[:2]
    target_w = 1500
    if w > target_w:
        scale = target_w / w
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        log.info(f"Downscaled image: {w}x{h} -> {img.shape[1]}x{img.shape[0]}")

    # 1. Auto-rotate if tilted
    try:
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_orig, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        if lines is not None:
            angles = []
            for line in lines[:30]:
                rho, theta = line[0]
                angle = theta * 180 / np.pi - 90
                if abs(angle) < 45: angles.append(angle)
            if angles:
                avg_angle = np.median(angles)
                if abs(avg_angle) > 1.0:
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), avg_angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                    log.info(f"Auto-rotated image by {avg_angle:.1f} degrees")
    except Exception as e:
        log.warning(f"Auto-rotation failed: {e}")

    variants = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Variant 0: Raw grayscale
    variants.append(gray)

    # 2. Deglossing
    deglossed = _remove_gloss(gray)

    # 3. Upscale for small text
    h, w = deglossed.shape[:2]
    target_ocr = 1500
    if w < target_ocr:
        scale = target_ocr / w
        deglossed = cv2.resize(deglossed, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Variant A: Deglossed + upscaled
    variants.append(deglossed)

    # Variant B: CLAHE contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(deglossed)

    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    contrast_sharp = cv2.filter2D(contrast, -1, sharpen_kernel)
    variants.append(contrast_sharp)

    # Variant C: Otsu threshold
    _, otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Variant D: Adaptive threshold
    denoised = cv2.bilateralFilter(contrast, 9, 75, 75)
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 6
    )
    variants.append(adaptive)

    # Variant E: Inverted (white text on dark)
    inverted = cv2.bitwise_not(adaptive)
    variants.append(inverted)

    return variants


# ── OCR (EasyOCR + Tesseract) ──────────────────────────────────────────────────

_EASY_READER = None
_EASY_LOCK = threading.Lock()

def _get_easyocr_reader():
    global _EASY_READER
    if _EASY_READER is not None:
        return _EASY_READER
    with _EASY_LOCK:
        if _EASY_READER is not None:
            return _EASY_READER
        try:
            import easyocr
            _EASY_READER = easyocr.Reader(['en'], gpu=False, verbose=False)
            log.info("EasyOCR reader loaded")
        except Exception as e:
            log.warning("EasyOCR not available: %s", e)
            _EASY_READER = None
        return _EASY_READER


def _run_easyocr(image_path: str) -> str:
    """Run EasyOCR on the image and return extracted text (with 60s timeout)."""
    reader = _get_easyocr_reader()
    if reader is None:
        return ""

    def _easyocr_job():
        img = cv2.imread(image_path)
        if img is None:
            return ""
        h, w = img.shape[:2]
        if w < 1000:
            scale = 1000.0 / w
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        results = reader.readtext(img, paragraph=True,
                                  contrast_ths=0.3, text_threshold=0.4,
                                  width_ths=0.7, height_ths=0.7)
        lines = []
        for detection in results:
            text = detection[1]
            if text.strip():
                lines.append(text.strip())
        return _fix_ocr_errors("\n".join(lines))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_easyocr_job)
            return future.result(timeout=60)
    except TimeoutError:
        log.warning("EasyOCR timed out after 60s — skipping")
        return ""
    except Exception as e:
        log.warning("EasyOCR failed: %s", e)
        return ""


def _run_tesseract_with_timeout(arr, config, timeout=30):
    """Run Tesseract with a per-call timeout."""
    def _job():
        return pytesseract.image_to_string(arr, config=config)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_job)
            return future.result(timeout=timeout)
    except TimeoutError:
        log.warning(f"Tesseract timed out after {timeout}s for config: {config}")
        return None
    except Exception:
        return None


def _run_ocr(image_path: str) -> str:
    """
    Run Tesseract + EasyOCR on preprocessed variants with region detection.
    Returns the best combined text from both engines.
    """
    variants = _preprocess_image_variants(image_path)
    if not variants:
        return ""

    configs = [
        "--psm 6 --oem 3",  # Sparse text (best for packaging)
        "--psm 3 --oem 3",  # Fully automatic
    ]
    
    best_text = ""
    log.info("Starting Multi-Pass OCR analysis (Tesseract + EasyOCR)...")
    
    # Pass 1: Try each config on each variant
    for v_idx, arr in enumerate(variants):
        for cfg_idx, cfg in enumerate(configs):
            try:
                text = _run_tesseract_with_timeout(arr, cfg, timeout=30)
                if text is None:
                    continue
                text_upper = text.upper()
                
                # EARLY EXIT: If we found enough high-quality text, stop
                if len(text.strip()) > 50:
                    has_expiry = any(k in text_upper for k in ["EXP", "BATCH", "B.NO", "MFG"])
                    has_medicine = any(k in text_upper for k in ["TABLET", "CAPSULE", "MG", "ML"])
                    
                    if has_expiry and has_medicine:
                        log.info(f"Excellent text found on variant {v_idx}, config {cfg_idx}. Exiting early.")
                        return _fix_ocr_errors(text)
                    
                    if len(text) > len(best_text):
                        best_text = text
                
                if len(text) > len(best_text):
                    best_text = text
            except Exception:
                continue

    # Pass 2: Region detection for small/dense text
    if "EXP" not in best_text.upper() or "BATCH" not in best_text.upper():
        log.info("Key fields missing in full OCR. Attempting region-based OCR...")
        best_variant = variants[1]
        regions = _detect_text_regions(best_variant)
        region_texts = []
        for x, y, w, h in regions[:15]:
            px, py = 5, 5
            roi = best_variant[max(0, y-py):min(best_variant.shape[0], y+h+py), 
                               max(0, x-px):min(best_variant.shape[1], x+w+px)]
            try:
                r_text = _run_tesseract_with_timeout(roi, "--psm 7 --oem 3", timeout=15)
                if r_text and len(r_text) > 3:
                    region_texts.append(r_text.strip())
            except:
                pass
        
        if region_texts:
            joined_regions = "\n".join(region_texts)
            if len(joined_regions) > 10:
                best_text += "\n" + joined_regions

    # Pass 3: EasyOCR fallback for difficult/better text extraction
    easy_text = _run_easyocr(image_path)
    if easy_text and len(easy_text) > len(best_text):
        log.info(f"EasyOCR found more text ({len(easy_text)} chars vs Tesseract {len(best_text)} chars)")
        best_text = easy_text
    elif easy_text and "EXP" in easy_text.upper() and "EXP" not in best_text.upper():
        log.info("EasyOCR found expiry date that Tesseract missed")
        best_text = best_text + "\n" + easy_text

    return _fix_ocr_errors(best_text)


def _fix_ocr_errors(text):
    """
    Fix common OCR substitutions and remove noise.
    Handles pharmaceutical-specific misreads.
    """
    if not text: return ""

    # 1. Correct common keyword mistakes
    kw_fixes = [
        (r'\b[Ff][Xx][Pp]\b', 'EXP'),
        (r'\b[Ee][Yy][Pp]\b', 'EXP'),
        (r'\b[Ee][Xx][Pp][Rr]\b', 'EXP'),
        (r'\b[Ee][Xx][Pp][Yy]\b', 'EXP'),
        (r'\b[Ee][Xx][Pp][\.\:]\b', 'EXP'),
        (r'\b[Ee][Xx]\.[Pp]\b', 'EXP'),
        (r'\b[Ee]\s*[Xx]\s*[Pp]\b', 'EXP'),
        (r'\b[Mm][Ff][Gg][Dd]?\b', 'MFG'),
        (r'\b[Bb]\.?[Nn][Oo]\.?\b', 'BATCH'),
        (r'\b[Bb]\s*[Nn][Oo]\b', 'BATCH'),
        (r'\b[LL][Oo][Tt]\b', 'BATCH'),
    ]
    for pattern, replacement in kw_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 2. Correct numeric substitutions in date-like contexts
    num_fixes = [
        (r'(?<=\d)[OoQq](?=[\d/\-\.])', '0'),
        (r'(?<=[\d/\-\.])[OoQq](?=\d)', '0'),
        (r'(?<=\d)[IlJj](?=[\d/\-\.])', '1'),
        (r'(?<=[\d/\-\.])[IlJj](?=\d)', '1'),
        (r'(?<=\d)[Ss](?=[\d/\-\.])', '5'),
        (r'(?<=\d)[Bb](?=[\d/\-\.])', '8'),
        (r'(?<=[\d/\-\.])[Bb](?=\d)', '8'),
    ]
    for pattern, replacement in num_fixes:
        text = re.sub(pattern, replacement, text)
    
    # 3. Clean noise but keep important characters
    text = re.sub(r'[^A-Za-z0-9\s\/\-\.\:\,]', ' ', text)
    
    # 4. Pharmaceutical specific common misreads
    text = text.replace('1P', 'IP') # Pharmacopoeia
    text = text.replace('08-2', '08-2') # Often part of a date
    text = text.replace('I0', '10')
    text = text.replace('O1', '01')
    
    text = re.sub(r' +', ' ', text)
    
    return text.strip()



# ── Field extraction ──────────────────────────────────────────────────────────

# Expiry date patterns (most → least specific)
_EXP_PATTERNS = [
    # Explicit keyword matches (High precision)
    r"(?:EXP(?:IRY|IRED?|\.)?|USE BEFORE|USE BY|BEST BEFORE|VALID UPTO|VALID TILL)[\s:=.\-]*DATE[\s:=.\-]*(\d{1,2}[/\-\.\s]+\d{1,2}[/\-\.\s]+\d{2,4})",
    r"(?:EXP(?:IRY|IRED?|\.)?|USE BEFORE|USE BY|BEST BEFORE|VALID UPTO|VALID TILL)[\s:=.\-]*(\d{1,2}[/\-\.\s]+\d{1,2}[/\-\.\s]+\d{2,4})",
    r"(?:EXP(?:IRY|IRED?|\.)?|USE BEFORE|USE BY|BEST BEFORE|VALID UPTO|VALID TILL)[\s:=.\-]*(\d{1,2}[/\-\.\s]+\d{2,4})",
    r"(?:EXP(?:IRY|IRED?|\.)?|USE BEFORE|USE BY|BEST BEFORE|VALID UPTO|VALID TILL)[\s:=.\-]*(\d{4}[/\-\.\s]+\d{1,2})",
    r"(?:EXP(?:IRY|IRED?|\.)?|USE BEFORE|USE BY|BEST BEFORE|VALID UPTO|VALID TILL)[\s:=.\-]*(\d{4})", # Use Before 2029
    r"(?:EXP(?:IRY|IRED?|\.)?|USE BEFORE|USE BY|BEST BEFORE|VALID UPTO|VALID TILL)[\s:=.\-]*((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s.\-/]?\d{2,4})",
    
    # New strict formats requested by USER
    r"EXP[:\s]*(\d{2}/\d{4})",     # EXP 12/2027
    r"EXP[:\s]*(\d{2}-\d{2})",     # EXP:08-26
    r"EXP[:\s]*(\d{2}/\d{2})",     # EXP:08/26
    r"E[:\s]*(\d{2}/\d{2})",       # E:08/26
    r"EXP[:\s]*([A-Z]{3,}\s?\d{2,4})", # EXP NOV 2027
    
    # Best before durations
    r"BEST BEFORE (\d+) MONTHS",
    r"USE WITHIN (\d+) MONTHS",
    
    # Generic date patterns (if no keyword found)
    r"\b(\d{1,2}[/\-\.\s]+\d{1,2}[/\-\.\s]+\d{2,4})\b",    # DD/MM/YYYY
    r"\b(\d{1,2}[/\-\.\s]+\d{4})\b",                    # MM/YYYY
    r"\b(\d{4}[/\-\.\s]+\d{1,2})\b",                    # YYYY/MM
    r"\b(\d{2}[/-]\d{2})\b",                            # MM/YY or MM-YY (e.g., 08-26)
    r"\b(\d{2}[/-]\d{4})\b",                            # MM-YYYY (e.g., 11-2027)
    r"\b((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s.\-/]?\d{2,4})\b",
]

_MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _is_exp_keyword(token):
    """Fuzzy-match EXP-like keywords and correct common OCR mistakes."""
    normalized = token.strip().upper().replace("×", "X")
    # Exact and common variants
    exact = {
        "EXP", "EXPIRY", "EXPIRED", "EXPDATE", "EXP DATE", "EXP.",
        "USE BEFORE", "USE BY", "BEST BEFORE",
        "VALID TILL", "VALID UNTIL", "VALID UPTO", "VALIDUPTO",
        "FXP", "EYP", "EX P", "EX.P", "E.X.P",
        "EXPY", "EXPR", "EXP-DATE",
    }
    if normalized in exact:
        return True

    # Tighter heuristic for OCR errors
    if len(normalized) >= 3 and len(normalized) <= 12:
        if normalized[0] in ("E", "F") and ("X" in normalized[:3] or "×" in normalized[:3]):
            return True
        if normalized.startswith("USE") and "BEFORE" in normalized:
            return True
        if normalized.startswith("BEST") and "BEFORE" in normalized:
            return True
        if normalized.startswith("VALID") and any(w in normalized for w in ("TILL", "UNTIL", "UPTO")):
            return True
    return False


def _normalize_date_format(date_str: str) -> str:
    """Standardize any date string to MM/YYYY format."""
    if not date_str or "Detected" in date_str: return date_str
    
    # Try to parse month name
    upper = date_str.upper()
    month = None
    for abbr, mon in _MONTH_ABBR.items():
        if abbr in upper:
            month = mon
            break
    
    year_m = re.search(r"(\d{2,4})", upper)
    year = int(year_m.group(1)) if year_m else None
    
    if month and year:
        if year < 100: year += 2000
        return f"{month:02d}/{year}"
        
    # Try numeric MM/YY or MM/YYYY
    nums = re.findall(r"\d+", date_str)
    if len(nums) >= 2:
        m_val = int(nums[0])
        y_val = int(nums[1])
        # Simple heuristic: MM is usually < 13
        if 1 <= m_val <= 12:
            m, y = m_val, y_val
        elif 1 <= y_val <= 12:
            m, y = y_val, m_val
        else:
            return date_str
            
        if y < 100: y += 2000
        return f"{m:02d}/{y}"
        
    return date_str


def _extract_expiry(text: str, image_path: str = None) -> Optional[str]:
    """Extract expiry date using robust line-by-line logic and pattern matching."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text_upper = text.upper()

    # Pass 1: Look for explicit patterns in the whole text
    for pat in _EXP_PATTERNS:
        matches = re.finditer(pat, text_upper)
        for match in matches:
            full_match = match.group(0)
            # Skip if it's a manufacturing date or storage instruction
            if re.search(r'\b(MFG|MFD|MANUFACTURED|MANUFACTURING|STORE|KEEP|BELOW|ABOVE|TEMPERATURE|TEMP)\b', full_match):
                continue
            
            candidate = match.group(1).strip()
            if _validate_date(candidate):
                return _normalize_date_format(candidate)

    # Pass 2: Look for EXP keyword and associated date on same/next line
    for i, line in enumerate(lines):
        line_upper = line.upper()
        if re.search(r'\b(MFG|MFD|MANUFACTURED|MANUFACTURING|STORE|KEEP|BELOW|ABOVE|TEMPERATURE|TEMP)\b', line_upper):
            continue

        if any(_is_exp_keyword(t) for t in re.split(r'[\s:.,|]+', line)):
            # Try same line
            date = _extract_date_from_text(line)
            if date: return _normalize_date_format(date)
            # Try next line
            if i + 1 < len(lines):
                date = _extract_date_from_text(lines[i + 1])
                if date: return _normalize_date_format(date)

    # Pass 3: LLM fallback (Deep inspection)
    try:
        from services.advanced_text_extraction import UnifiedAIExtractor
        llm_res = UnifiedAIExtractor.extract_details(text, image_path=image_path)
        if llm_res.get("expiry_date"):
            return _normalize_date_format(llm_res["expiry_date"])
    except Exception as e:
        log.warning(f"LLM extraction failed: {e}")

    return None


def _extract_date_from_text(text):
    """Specific extractor for dates within a text line."""
    # Pre-clean the line for common OCR number errors
    text = text.upper().replace('I', '1').replace('L', '1').replace('O', '0').replace('S', '5').replace('B', '8')
    
    patterns = [
        r'(\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{4})',    # DD/MM/YYYY
        r'(\d{1,2}[/\-.\s]\d{1,2}[/\-.\s]\d{2})',    # DD/MM/YY
        r'((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s.\-/]?\d{2,4})',
        r'(\d{1,2}[/\-.\s]\d{4})',               # MM/YYYY
        r'(\d{1,2}[/\-.\s]\d{2})',               # MM/YY
        r'(\d{2}[/\-]\d{2})',                    # 08/26
        r'([A-Z]{3,}\s?\d{2,4})',                # JUN2026
        r'(\d{4})',                              # 2029
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if _validate_date(candidate):
                return candidate
    return None


def _validate_date(s):
    if not s: return False
    # Clean up common OCR noise in dates
    s = s.strip().replace(' ', '/')
    s = re.sub(r'[\\|]', '/', s)
    s = re.sub(r'//+', '/', s)
    
    try:
        # DD/MM/YYYY or similar
        if re.match(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}$', s):
            p = re.split(r'[/\-\.]', s)
            if len(p) == 3:
                v1, v2, v3 = int(p[0]), int(p[1]), int(p[2])
                # DD/MM/YYYY
                if 2020 <= v3 <= 2050 and 1 <= v2 <= 12 and 1 <= v1 <= 31:
                    return True
                # YYYY/MM/DD
                if 2020 <= v1 <= 2050 and 1 <= v2 <= 12 and 1 <= v3 <= 31:
                    return True
        
        # MM/YYYY or YYYY/MM
        if re.match(r'^\d{1,4}[/\-\.]\d{1,4}$', s):
            p = re.split(r'[/\-\.]', s)
            v1, v2 = int(p[0]), int(p[1])
            # MM/YYYY
            if 1 <= v1 <= 12 and 2020 <= v2 <= 2050: return True
            # YYYY/MM
            if 2020 <= v1 <= 2050 and 1 <= v2 <= 12: return True
            # MM/YY
            if 1 <= v1 <= 12 and 20 <= v2 <= 50: return True
            # YY/MM
            if 20 <= v1 <= 50 and 1 <= v2 <= 12: return True

        # YYYY format (e.g., Use Before 2029)
        if re.match(r'^\d{4}$', s):
            year = int(s)
            return 2020 <= year <= 2050

        # Month name formats
        s_upper = s.upper()
        for abbr in ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]:
            if abbr in s_upper:
                # Ensure it contains a year-like number
                return bool(re.search(r'\d{2,4}', s))
    except: pass
    return False


def _parse_expiry_status(exp_str: Optional[str]) -> Optional[str]:
    """Return 'EXPIRED', 'VALID', or None."""
    if not exp_str:
        return None
    upper = exp_str.upper()
    # Month abbreviation format: JAN/2025
    for abbr, mon in _MONTH_ABBR.items():
        if abbr in upper:
            year_m = re.search(r"\d{2,4}", upper)
            if year_m:
                year = int(year_m.group())
                if year < 100:
                    year += 2000
                try:
                    dt = datetime(year, mon, 1)
                    return "EXPIRED" if dt < datetime.now() else "VALID"
                except ValueError:
                    pass
    # Numeric MM/YYYY
    m = re.match(r"^(\d{1,2})[\/\-\.](\d{4})$", exp_str)
    if m:
        try:
            dt = datetime(int(m.group(2)), int(m.group(1)), 1)
            status = "EXPIRED" if dt < datetime.now() else "VALID"
            log.info(f"Parsed numeric MM/YYYY: {exp_str} -> {dt.strftime('%Y-%m')} -> {status}")
            return status
        except ValueError:
            pass
    # Numeric MM/YY
    m = re.match(r"^(\d{1,2})[\/\-\.](\d{2})$", exp_str)
    if m:
        year = 2000 + int(m.group(2))
        try:
            dt = datetime(year, int(m.group(1)), 1)
            status = "EXPIRED" if dt < datetime.now() else "VALID"
            log.info(f"Parsed numeric MM/YY: {exp_str} -> {dt.strftime('%Y-%m')} -> {status}")
            return status
        except ValueError:
            pass
    return "UNKNOWN"


# ── Medicine Name Extraction ──────────────────────────────────────────────────

_SKIP_PATTERNS = [
    r'^(mg|ml|tablet|capsule|cap|tab|strip|pack|box|sachet|per|caps|gm|mcg|iu)\s*$',
    r'^(exp|expiry|use.?before|mfd|mfg|batch|bn|lot|www|http|mrp|rs|inclusive|lic|reg|tm|r|c|gm|sachet|dosage|composition)',
    r'^(store|keep|protect|cool|dry|place|temperature|below|above|degree|reach|children|outer|inner|warning|caution|note)',
    r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$',   # pure date
    r'^[A-Z0-9]{5,20}$',                        # batch numbers
    r'^\d+\s*(mg|ml|mcg|iu|cap|caps|tablet|tablets)$', # dose or count only
    r'^rs\.?\s*\d+',                             # price
    r'^\d{4,}$',                                 # long number
    r'^(each|contains|ingredients?|storage|schedule|drug|side|effect|direction|prescription|pharmacist)',
    r'^\d+\s*[x×]\s*\d+',                       # dimension like 10x10
    r'^(mfg|mfd|manuf)',
    r'^(see|side|effect|direction|dosage|storage|keep|reach|children)',
    r'^[0-9\s/.\-:]+$', # pure noise/date
    r'^[A-Z]{1,2}$',    # single letters
]

def _extract_medicine_name_raw(text: str, image_path: str = None) -> Optional[str]:
    """Find the medicine name using line-by-line rules and prominent text detection."""
    if not text: return None
    
    # ── CLEANING & PREPARATION ──
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 2]
    text_upper = text.upper()
    
    # Priority 0: Explicit check for known high-value names
    high_value = [
        "DERIPHYLLIN", "ITROSTRED", "CIP-ZOX", "CIPZOX", "CALPOL", "DOLO", 
        "PANTOP", "AUGMENTIN", "CROCIN", "AZITHRAL", "LIMCEE", "ZIFI", 
        "TAXIM", "TELMA", "AMLOKIND", "AMLODIPINE", "PANTOPRAZOLE", 
        "PARACETAMOL", "METFORMIN", "AMLOVAS", "AMKIND", "ATARAX", "GLYCOMET",
        "PANTOCID", "PAN-D", "RABIPRAZOLE", "VOGLIBOSE", "TELMIKIND", "TELVAS",
        "MEFTAL", "COMBIFLAM", "VOVERAN", "CITRIZINE", "OKACET", "LIV-52"
    ]
    for hv in high_value:
        if hv in text_upper:
            brand_dosage_re = re.compile(rf'\b({hv}(?:\s[A-Z0-9]+)*[\s\-]\d+(?:\s?(?:MG|ML|MCG|IU))?)\b', re.IGNORECASE)
            m = brand_dosage_re.search(text)
            if m: return m.group(1).strip()
            return hv

    # Priority 1: Brand name with dosage pattern (e.g., PARACIP-500, DOLO 650)
    dosage_re = re.compile(r'\b([A-Z]{3,}(?:\s[A-Z]+)*[\s\-]\d+(?:\s?(?:MG|ML|MCG|IU))?)\b', re.IGNORECASE)
    for line in lines[:15]:
        if any(re.search(p, line, re.IGNORECASE) for p in _SKIP_PATTERNS[1:3]):
            continue
        m = dosage_re.search(line)
        if m:
            name = re.sub(r'[^A-Za-z0-9\s\-\+\.]', '', m.group(1)).strip()
            if len(name) >= 4:
                return name

    # Priority 2: Frequent Word Analysis (Medicine names often repeat on strips)
    tokens = re.findall(r'\b[A-Z]{4,}\b', text_upper)
    filtered_tokens = [t for t in tokens if not any(re.match(p, t, re.IGNORECASE) for p in _SKIP_PATTERNS)]
    if filtered_tokens:
        from collections import Counter
        counts = Counter(filtered_tokens)
        for word, count in counts.most_common(3):
            if count >= 2:
                for line in lines:
                    if word in line.upper() and re.search(r'\d+\s?(?:MG|ML|MCG|IU)', line.upper()):
                         return word
                return word

    # Priority 3: First non-skipped prominent line that looks like a brand name
    candidates = []
    for line in lines[:10]:
        if sum(c.isdigit() for c in line) > len(line) * 0.4: continue
        if any(re.match(p, line, re.IGNORECASE) for p in _SKIP_PATTERNS): continue
        clean_line = re.sub(r'[^A-Z0-9\s\-]', '', line.upper()).strip()
        if len(clean_line) > 3 and not _is_garbage(clean_line):
            # Prefer lines with both letters and numbers (like DOLO-650)
            score = 0
            if re.search(r'[A-Z]{3,}[\s\-]\d+', clean_line): score += 3
            if re.search(r'[A-Z]{4,}', clean_line): score += 2
            if re.search(r'[A-Z][a-z]', line): score += 1  # mixed case suggests proper name
            if not re.search(r'(TABLET|CAPSULE|STRIP|PACK|SYRUP|CREAM|LOTION|DROP|INJECTION)', clean_line):
                score += 1
            candidates.append((score, clean_line))

    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]

    # Fallback: Visual AI-based name extraction
    try:
        from services.advanced_text_extraction import UnifiedAIExtractor
        llm_res = UnifiedAIExtractor.extract_details(text, image_path=image_path)
        ai_name = llm_res.get("medicine_name")
        if ai_name and not _is_garbage(ai_name):
            return ai_name
    except Exception as e:
        log.warning(f"LLM fallback for name failed: {e}")

    return None


def _is_ocr_failed(text: str) -> bool:
    """
    Detect if OCR completely failed - text is mostly garbage/unintelligible.
    Returns True if OCR should not be trusted and we should use AI fallback.
    """
    if not text or len(text.strip()) < 10:
        return True
    
    # Count different types of characters
    alpha_count = sum(1 for c in text if c.isalpha())
    digit_count = sum(1 for c in text if c.isdigit())
    space_count = sum(1 for c in text if c.isspace())
    other_count = len(text) - alpha_count - digit_count - space_count
    
    total_meaningful = alpha_count + digit_count
    if total_meaningful == 0:
        return True  # No text at all
    
    # If more than 50% is special characters (OCR garbage), it failed
    if other_count / len(text) > 0.5:
        return True
    
    # Check for excessive repeating patterns (OCR noise like "eeeeee" or "PPP")
    repeating = len(re.findall(r'(.)\1{2,}', text))
    if repeating > len(text) * 0.05:  # >5% repeating chars is suspicious
        return True
    
    # Check if the text looks like it should have keywords but doesn't
    # (medicine labels usually have EXP, BATCH, MG, etc.)
    text_upper = text.upper()
    has_key_words = any(k in text_upper for k in [
        "EXP", "BATCH", "B.N", "LOT", "MFG", "MG", "ML", "TABLET", "CAPSULE"
    ])
    
    # If text is long but has no key identifiers, OCR probably failed
    if len(text) > 100 and not has_key_words:
        return True
    
    return False


def _is_garbage(text: str) -> bool:
    """Detect if OCR text is likely garbage/nonsense."""
    if not text: return True
    upper = text.strip().upper()
    
    # Generic pharmaceutical labels that aren't product names
    SKIP_LABELS = [
        "TABLETS", "CAPSULES", "SYRUP", "INJECTION", "STRIP OF", "DMF", "API GRADE",
        "USP", "BP", "IP", "PHARMA", "LABORATORIES", "LIMITED", "LTD", "MFG", "EXP",
        "TAB", "CAP", "SACHET", "VIAL", "DROP", "LOTION", "CREAM", "OINTMENT", "GEL",
        "SEE", "SEES", "SIDE", "EFFECTS", "KEEP", "STORE", "REACH", "CHILDREN",
        "WARNING", "CAUTION", "SCHEDULE", "PRESCRIPTION", "PHARMACIST",
        "DOSAGE", "COMPOSITION", "EACH", "CONTAINS", "INGREDIENTS",
        "JUL", "JUN", "AUG", "SEP", "OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY",
        "ETE", "ETT", "EII", "III", "LL", "OOO", "ZZZ", "XXX", "IEEE", "EEEE", "IIII", "OOOO", "UUUU",
        "WOR", "WOS", "OSE", "WOR OS EE", "WOROS"  # Common OCR mistakes
    ]
    if upper in SKIP_LABELS: return True
    if upper.strip("0123456789 ").strip() in SKIP_LABELS: return True

    # ── PATTERN-BASED GARBAGE ──
    
    # Catch names that look like truncated dates (e.g. JUN 2027 misread as JUN 2)
    if re.search(r'^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|ETE|ETT|EII)\s*\d{1,2}$', upper):
        return True
    if len(upper) < 3: return True
    
    # Pattern: Short sequences of capitalized letters with few vowels (e.g. "WOR OS EE")
    # But exclude numeric words and abbreviations
    words = upper.split()
    if len(words) <= 3:
        short_vowel_words = 0
        for w in words:
            # Skip abbreviations and numeric
            if w in ["MFG", "EXP", "LOT", "BN", "ML", "MG", "MCG", "IU"]: continue
            if re.match(r'^\d', w): continue  # Starts with digit
            # Count legitimate words with no vowels
            if len(w) <= 3 and not re.search(r'[AEIOUY]', w):
                short_vowel_words += 1
        
        if short_vowel_words >= 2 and short_vowel_words >= len(words) * 0.5:
            return True

    # Strip common pharmaceutical suffixes/prefixes for checking
    stripped = re.sub(r'\s*\d+\s*(MG|ML|MCG|IU|TAB|CAP|TABLET|CAPSULE)?\s*$', '', upper).strip()
    stripped = re.sub(r'^(TAB|CAP|TABLET|CAPSULE|STRIP|PACK)\s+', '', stripped).strip()
    
    # Storage/instruction text
    if any(k in upper for k in [
        "STORE BELOW", "STORE AT", "KEEP IN", "COOL DRY", "PROTECT FROM",
        "REACH OF CHILDREN", "EACH", "CONTAINS", "COMPOSITION", "INGREDIENT",
        "DIRECTION", "DOSAGE", "WARNING", "CAUTION", "SCHEDULE",
        "PRESCRIPTION", "PHARMACIST", "MANUFACTURED", "MARKETED",
    ]):
        return True

    if not re.search(r'[A-Z]', upper): return True

    # Pronounceability check: Pharmaceutical names usually have a reasonable vowel ratio
    vowels = re.findall(r'[AEIOUY]', upper)
    if len(upper) > 4:
        vowel_ratio = len(vowels) / len(upper)
        if vowel_ratio < 0.15: return True  # Too few vowels
        
    # Word-level garbage check: words >= 3 chars with no vowels are likely noise
    for word in upper.split():
        # Skip common abbreviations, suffixes, and numeric strings
        if word in ["MFG", "EXP", "LOT", "BN", "ML", "MG", "MCG", "IU"]: continue
        if re.match(r'^\d+.*$', word): continue  # Skip numeric strings like "500" or "400MG"
        if re.match(r'^.*\d+$', word): continue  # Skip alpha-numeric like "MG400"
        if len(word) >= 3 and not re.search(r'[AEIOUY]', word):
            return True # e.g. "WRF", "CTN", "BIP", "WOR" (misread)
    
    # Check for excessive repeating characters (common OCR noise)
    # But be lenient - some repeats are okay
    if re.search(r'(.)\1\1\1', upper): return True # e.g. "AAAA"
    if re.search(r'([A-Z])\1\s([A-Z])\2\s([A-Z])\3', upper): return True # e.g. "EE EE EE"
    
    # Check for garbage patterns like "EE EE 00" (space-separated short pairs)
    if re.match(r'^[A-Z]{2}\s+[A-Z]{2}(?:\s+[A-Z0-9]{2})?$', upper):
        # This matches things like "WOR OS EE" or "EE EE 00"
        return True

    # Too many strange symbols
    symbols = re.findall(r'[^A-Za-z0-9\s\-]', text)
    if len(symbols) > len(text) * 0.3: return True

    # Too short after cleaning
    clean = re.sub(r'[^A-Z]', '', upper)
    if len(clean) < 3:
        return True
    
    return False


# ── Multi-Pass Fuzzy Matching ─────────────────────────────────────────────────

_FUZZY_THRESH_1A = 88   # Pass 1A: strict token sort ratio
_FUZZY_THRESH_1B = 75   # Pass 1B: partial ratio
_FUZZY_THRESH_2  = 65   # Pass 2 : token set ratio (relaxed)

def _fuzzy_match_name(query: str,
                      candidates: list[str],
                      return_best: bool = False
                      ) -> tuple[bool, float, str]:
    """
    Multi-pass fuzzy matching against a list of candidate strings.

    Pass 1A → token_sort_ratio  ≥ _FUZZY_THRESH_1A
    Pass 1B → partial_ratio     ≥ _FUZZY_THRESH_1B
    Pass 2  → token_set_ratio   ≥ _FUZZY_THRESH_2

    Returns (matched: bool, best_score: float, best_candidate: str).
    """
    if not query or not candidates:
        return False, 0.0, ""
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        log.warning("rapidfuzz not installed — fuzzy matching disabled.")
        return False, 0.0, ""

    # Pass 1A
    res1a = process.extractOne(query, candidates,
                               scorer=fuzz.token_sort_ratio)
    if res1a and res1a[1] >= _FUZZY_THRESH_1A:
        return True, res1a[1], res1a[0]

    # Pass 1B
    res1b = process.extractOne(query, candidates,
                               scorer=fuzz.partial_ratio)
    if res1b and res1b[1] >= _FUZZY_THRESH_1B:
        return True, res1b[1], res1b[0]

    # Pass 2
    res2 = process.extractOne(query, candidates,
                              scorer=fuzz.token_set_ratio)
    if res2 and res2[1] >= _FUZZY_THRESH_2:
        return True, res2[1], res2[0]

    # Return best overall even if unmatched (for logging)
    best = max([r for r in [res1a, res1b, res2] if r],
               key=lambda x: x[1], default=None)
    if best:
        return False, best[1], best[0]
    return False, 0.0, ""


def _fuzzy_match_company(query: str,
                         candidates: list[str]) -> tuple[bool, float, str]:
    """Dedicated company matching (identical pass logic, separate thresholds)."""
    return _fuzzy_match_name(query, candidates)


# ── T(I): Text-layer verification ─────────────────────────────────────────────

def _text_layer(norm_text: str) -> dict:
    """
    Simplified text layer (Database matching removed).
    Returns basic structure for compatibility.
    """
    return {
        "name_matched":    False,
        "name_score":      0.0,
        "matched_name":    "",
        "company_matched": False,
        "company_name":    "",
    }


# ── A(I): CNN classification ──────────────────────────────────────────────────

def _cnn_layer(image_path: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """
    Returns (cnn_status, confidence_pct, confidence_label) or (None, None, None).

    cnn_status one of: 'real', 'fake', 'likely_real', 'likely_fake', 'uncertain'
    """
    mdl = _load_model()
    if mdl is None:
        return None, None, None
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None, None

        img_r = cv2.resize(img, (128, 128))
        img_n = img_r.astype(np.float32) / 255.0
        img_b = np.expand_dims(img_n, axis=0)

        pred = mdl.predict(img_b, verbose=0)

        if pred.ndim == 2 and pred.shape[1] == 1:
            raw_prob = float(pred[0][0])
        elif pred.ndim == 2 and pred.shape[1] >= 2:
            raw_prob = float(pred[0][1])
        else:
            flat = pred.flatten()
            raw_prob = float(flat[0]) if flat.size else 0.0
        
        log.info(f"CNN Raw Probability: {raw_prob:.4f}")
        # print(f"[DEBUG] CNN Raw Prob: {raw_prob:.4f}")

        # Inverted logic per user report: REAL images were showing as FAKE
        # Corrected logic: Prob > 0.5 is class 1 (REAL), Prob <= 0.5 is class 0 (FAKE)
        if raw_prob > 0.5:
            pred_class = "real"
            conf_val = raw_prob
        else:
            pred_class = "fake"
            conf_val = 1.0 - raw_prob

        pct = conf_val * 100

        if pct < 65:
            status, label = "uncertain", "UNCERTAIN"
        elif pct < 85:
            status = f"likely_{pred_class}"
            label  = f"LIKELY {pred_class.upper()}"
        else:
            status = pred_class
            label  = pred_class.upper()

        # Backward compatibility for 'confidence' value (app expects 0-100 score of the winning class)
        return status, pct, label

    except Exception as exc:
        log.error("CNN prediction error: %s", exc)
        return None, None, None


# ── 7-Pass Verdict Logic ──────────────────────────────────────────────────────

def _verdict(
    cnn_status:      str,
    database_matched: bool,
    name_found:      bool,
    expiry_detected: bool = False,
) -> str:
    """
    Real-World Binary Verdict Logic:
    1. REAL: Database Match (Highest Priority)
    2. REAL: CNN says Real/Likely Real AND a valid name was found (even if not in DB)
    3. REAL: CNN Uncertain AND both valid Name + Expiry were found
    4. FAKE: CNN says Fake/Likely Fake AND no database match
    5. FAKE: Otherwise
    """
    # 1. Database Match - Definite Real
    if database_matched:
        return "REAL"

    # 2. CNN Confidence - Trust the trained model as primary source
    is_cnn_real = cnn_status in ("real", "likely_real")
    is_cnn_fake = cnn_status in ("fake", "likely_fake")
    
    if is_cnn_real:
        return "REAL"
    
    if is_cnn_fake and not database_matched:
        return "FAKE"
    
    # 3. CNN Uncertain / No Model - Trust OCR evidence
    if cnn_status in ("uncertain", "UNCERTAIN (NO MODEL)"):
        if expiry_detected:
            return "REAL"
    
    # 4. No CNN model available at all — trust OCR evidence
    if cnn_status is None:
        if expiry_detected or name_found:
            return "REAL"
    
    return "FAKE"


# ── Expiry-based CNN override ─────────────────────────────────────────────────

def _apply_expiry_override(result: dict) -> dict:
    """If medicine is expired, override status to 'FAKE'."""
    if result.get("expiry_status") == "EXPIRED":
        result["verdict"] = "FAKE"
        result["status"] = "FAKE"
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def predict_image(image_path: str) -> dict:
    """
    Full 3-layer pipeline: A(I) + T(I) + C(I) → V

    Returns a dict compatible with app.py:
      medicine_name, expiry_date, extracted_text,
      expiry_status, verdict, company_name
    """
    result = {
        "status":           "detection_failed",
        "confidence":       None,
        "confidence_label": None,
        "medicine_name":    None,
        "expiry_date":      None,
        "batch_number":     None,
        "extracted_text":   "",
        "expiry_status":    None,
        "verdict":          "UNKNOWN",
        "company_name":     None,
        "ocr_confidence":   0.0,
    }
    exp_str = None

    try:
        # ── Parallel Execution: Layer A (CNN) + Layer T (OCR) ──────────────────
        # Running these in parallel saves 2-5 seconds on CPU
        with ThreadPoolExecutor(max_workers=2) as executor:
            cnn_future = executor.submit(_cnn_layer, image_path)
            ocr_future = executor.submit(_run_ocr, image_path)
            
            # Wait for both to complete
            cnn_status, confidence, conf_label = cnn_future.result()
            raw_text = ocr_future.result()

        if cnn_status is None:
            # CNN unavailable — fall back to OCR-only path with neutral status
            cnn_status  = "uncertain"
            confidence  = 50.0
            conf_label  = "UNCERTAIN (NO MODEL)"

        result["status"]           = cnn_status
        result["confidence"]       = confidence
        result["confidence_label"] = conf_label
        result["extracted_text"]   = raw_text

        norm_text = _normalize(raw_text)

        # ── CHECK FOR OCR FAILURE ──────────────────────────────────────────────────
        # If OCR completely failed, skip to AI immediately
        ocr_failed = _is_ocr_failed(raw_text)
        if ocr_failed:
            log.warning("OCR quality check: Text appears to be mostly garbage. Using AI vision directly.")
            try:
                from services.advanced_text_extraction import UnifiedAIExtractor
                deep_ai = UnifiedAIExtractor.extract_details(raw_text, image_path=image_path)
                
                if deep_ai and isinstance(deep_ai, dict):
                    ai_name = deep_ai.get("medicine_name")
                    ai_expiry = deep_ai.get("expiry_date")
                    
                    if ai_name:
                        result["medicine_name"] = ai_name.upper()
                        log.info(f"AI extracted medicine name: {result['medicine_name']}")
                    if ai_expiry:
                        result["expiry_date"] = _normalize_date_format(ai_expiry)
                        result["expiry_status"] = _parse_expiry_status(result["expiry_date"])
                        log.info(f"AI extracted expiry date: {result['expiry_date']}")
            except Exception as e:
                log.warning(f"AI fallback for failed OCR: {e}")

        # Extract expiry (uses Gemini fallback internally if not already set)
        if not result.get("expiry_date"):
            exp_str = _extract_expiry(raw_text, image_path=image_path)
            result["expiry_date"]   = exp_str
            result["expiry_status"] = _parse_expiry_status(exp_str)
        else:
            result["expiry_status"] = _parse_expiry_status(result.get("expiry_date"))

        # ── Layer T + C: fuzzy matching + company check ─────────────
        # (Skip if we already have good AI results)
        if not result.get("medicine_name") or ocr_failed:
            tl = _text_layer(norm_text)

            matched_name    = tl["matched_name"]
            name_matched    = tl["name_matched"]
            company_matched = tl["company_matched"]
            company_name    = tl["company_name"]

            # Set medicine_name: prefer fuzzy-matched canonical name, else raw extractor
            if name_matched and matched_name:
                result["medicine_name"] = matched_name.upper()
            else:
                # Fallback to robust raw extractor with visual AI
                if not ocr_failed:  # Only use if OCR wasn't completely bad
                    raw_name = _extract_medicine_name_raw(raw_text, image_path=image_path)
                    if raw_name:
                        result["medicine_name"] = raw_name.upper()
                    else:
                        # Last resort: first long token
                        tokens = [t for t in norm_text.split() if len(t) > 3]
                        result["medicine_name"] = tokens[0].upper() if tokens else None

            result["company_name"]  = company_name.title() if company_name else None
        else:
            result["company_name"] = None

        # ── Handle Missing Extraction (Smart Fallback) ─────────────────────────────
        # Trigger AI if missing fields OR if the name looks like garbage/nonsense
        is_name_garbage = _is_garbage(result.get("medicine_name", ""))
        
        needs_ai = (
            not result.get("medicine_name") or 
            result["medicine_name"] in ["Not Detected", "N/A"] or 
            is_name_garbage or
            not result.get("expiry_date") or 
            result.get("expiry_status") in ["UNKNOWN", None] or
            len(result.get("medicine_name", "")) < 4
        )

        if needs_ai:
            log.info(f"Low confidence (Garbage={is_name_garbage}, OCRFailed={ocr_failed}). Triggering AI inspection...")
            try:
                from services.advanced_text_extraction import UnifiedAIExtractor
                deep_ai = UnifiedAIExtractor.extract_details(raw_text, image_path=image_path)
                
                if deep_ai and isinstance(deep_ai, dict):
                    # Update name if missing or garbage
                    ai_name = deep_ai.get("medicine_name")
                    if ai_name and (result["medicine_name"] == "Not Detected" or is_name_garbage):
                        result["medicine_name"] = ai_name.upper()
                        log.info(f"AI fixed medicine name: {result['medicine_name']}")
                        
                    # Update expiry if missing or unknown
                    ai_expiry = deep_ai.get("expiry_date")
                    if ai_expiry and (not result.get("expiry_date") or result["expiry_status"] == "UNKNOWN"):
                        result["expiry_date"] = _normalize_date_format(ai_expiry)
                        result["expiry_status"] = _parse_expiry_status(result["expiry_date"])
                        exp_str = result["expiry_date"]
                        log.info(f"AI fixed expiry date: {result['expiry_date']}")
            except Exception as e:
                log.warning(f"AI inspection fallback failed: {e}")

        # Extract Batch Number
        batch_m = re.search(r'(?:BATCH|B\.?NO|LOT|BN)[\s:=.\-]*([A-Z0-9\-\/]{4,20})', raw_text.upper())
        result["batch_number"] = batch_m.group(1).strip() if batch_m else "Not Detected"

        # Calculate OCR Confidence
        ocr_conf = 0.0
        if result["medicine_name"] and result["medicine_name"] != "Not Detected": ocr_conf += 40.0
        if result["expiry_date"] and result["expiry_date"] != "Expiry Date Not Clearly Visible": ocr_conf += 30.0
        if result["batch_number"] and result["batch_number"] != "Not Detected": ocr_conf += 20.0
        if name_matched: ocr_conf += 10.0
        result["ocr_confidence"] = min(ocr_conf, 100.0)

        # Final sanity check on medicine name for display
        if _is_garbage(result.get("medicine_name", "")):
            result["medicine_name"] = "Not Detected"

        found_any_name = bool(result.get("medicine_name") and result["medicine_name"] != "Not Detected" and not _is_garbage(result["medicine_name"]))

        log.info(f"Verdict Diagnostics | CNN: {cnn_status} | Name: {result['medicine_name']} | Expiry: {result['expiry_date']}")
        
        result["verdict"] = _verdict(
            cnn_status       = cnn_status,
            database_matched = False, # CSV Database removed
            name_found       = found_any_name,
            expiry_detected  = bool(exp_str),
        )

        # Expiry override on final status
        old_verdict = result["verdict"]
        result = _apply_expiry_override(result)
        if old_verdict == "REAL" and result["verdict"] == "FAKE":
            log.info("Verdict overridden to FAKE due to EXPIRED status.")

        # Final cleanup for display
        if not result.get("medicine_name"):
            result["medicine_name"] = "Not Detected"
        
        if not result.get("expiry_date") or result["expiry_date"] in ["Expiry Not Detected", "N/A"]:
            result["expiry_date"] = "Expiry Date Not Clearly Visible"
            result["expiry_status"] = "UNKNOWN"
            
        if "FAKE" in result["verdict"] or "SUSPICIOUS" in result["verdict"]:
            result["expiry_status"] = "INVALID (FAKE)"
            # result["expiry_date"] = "N/A" # User wants actual expiry if detected, so we keep it if found
            result["batch_no"] = "N/A" # Hide batch for fake products
            result["confidence_label"] = "FAKE"
            result["status"] = "FAKE"
            log.info("Fake/Suspicious product detected.")

        log.info(
            "predict_image | status=%s | verdict=%s | name=%s | company=%s",
            result["status"], result["verdict"],
            result["medicine_name"], result["company_name"],
        )

    except Exception as exc:
        log.error("predict_image error: %s", exc, exc_info=True)

    return result


def run_cnn(image_path: str):
    """Backward-compat shim — returns (status, confidence_pct, label)."""
    return _cnn_layer(image_path)


def get_confidence(image_path: str):
    """Return confidence score for the image (backward-compat)."""
    res = predict_image(image_path)
    return res["confidence"] if res["confidence"] is not None else "N/A"
