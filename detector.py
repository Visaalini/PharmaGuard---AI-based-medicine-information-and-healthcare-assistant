"""
PHARMAGUARD - OCR-based Medical Label Analyzer
================================================
Extracts ONLY:
  1. Medicine Name
  2. Expiry Date

Rules:
  - Ignore batch number, MFG date, price, address, warnings
  - Handle blurry text
  - Fix OCR mistakes: 'O' -> '0', 'I' -> '1' in numeric contexts
  - Fuzzy EXP keyword detection (Fxp, Bxp, E×p, etc.)
  - Supported date formats: MM/YY, MM/YYYY, DD/MM/YYYY, Month YYYY
"""

import re
import json
import cv2
import numpy as np
import pytesseract

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

print("[INFO] Detector loaded (pytesseract backend)")


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_image(img):
    """
    Produce multiple enhanced grayscale variants for best OCR accuracy.
    Returns list of numpy arrays.
    """
    variants = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale narrow images for better OCR
    h, w = gray.shape[:2]
    if w < 1500:
        scale = 1500.0 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Bilateral denoise
    denoised = cv2.bilateralFilter(contrast, 11, 95, 95)

    # Otsu threshold
    _, otsu = cv2.threshold(denoised, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    variants.extend([contrast, denoised, otsu, adaptive])
    return variants


# ─────────────────────────────────────────────────────────────────────────────
# OCR RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def _run_tesseract(arr):
    """Run pytesseract on a preprocessed grayscale array. Returns raw text."""
    # psm 6 = assume a uniform block of text; psm 11 = sparse text
    configs = [
        "--psm 6 --oem 3",
        "--psm 11 --oem 3",
        "--psm 4 --oem 3",
    ]
    best = ""
    for cfg in configs:
        try:
            text = pytesseract.image_to_string(arr, config=cfg)
            if len(text) > len(best):
                best = text
        except Exception:
            pass
    return best


def _ocr_all_variants(preprocessed):
    """Run OCR on every variant, return the longest result."""
    best_text = ""
    for arr in preprocessed:
        text = _run_tesseract(arr)
        if len(text) > len(best_text):
            best_text = text
    return best_text


# ─────────────────────────────────────────────────────────────────────────────
# OCR ERROR CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def _fix_ocr_errors(text):
    """
    Fix common OCR substitutions in numeric contexts:
      'O' -> '0', 'I' -> '1', 'l' -> '1', 'S' -> '5'
    Only applied inside number-like sequences (next to digits or / - ).
    """
    # Replace common OCR errors in date-like patterns
    fixes = [
        (r'(?<=\d)[OoQq](?=[\d/\-])', '0'),
        (r'(?<=[\d/\-])[OoQq](?=\d)', '0'),
        (r'(?<=\d)[Il](?=[\d/\-])', '1'),
        (r'(?<=[\d/\-])[Il](?=\d)', '1'),
        (r'(?<=\d)[Ss](?=[\d/\-])', '5'),
    ]
    for pattern, replacement in fixes:
        text = re.sub(pattern, replacement, text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# EXP KEYWORD DETECTION  (fuzzy – handles OCR misreads)
# ─────────────────────────────────────────────────────────────────────────────

_EXP_EXACT = {
    "EXP", "EXPIRY", "EXPIRED", "EXPDATE", "EXP DATE", "EXP.",
    "USE BEFORE", "USE BY", "BEST BEFORE",
    "VALID TILL", "VALID UNTIL", "VALID UPTO",
}


def _is_exp_keyword(token):
    """
    Fuzzy-match EXP-like keywords even with OCR noise.
    Catches: Fxp, Bxp, E×p, exp, EXPIRY, Use Before, Valid Till, …
    """
    normalized = token.strip().upper().replace("×", "X").replace("×", "X")
    if normalized in _EXP_EXACT:
        return True
    # Single-token fuzzy: starts with E/F/B/P and contains X/Y/P in first 3 chars
    if (len(normalized) >= 2
            and normalized[0] in ("E", "F", "B", "P")
            and any(c in normalized[:3] for c in ("X", "Y", "P"))):
        return True
    if normalized.startswith("USE") and "BEFORE" in normalized:
        return True
    if normalized.startswith("BEST") and "BEFORE" in normalized:
        return True
    if normalized.startswith("VALID") and any(
            w in normalized for w in ("TILL", "UNTIL", "UPTO")):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# DATE PATTERNS  (ordered: most specific first)
# ─────────────────────────────────────────────────────────────────────────────

_DATE_PATTERNS = [
    # DD/MM/YYYY  or  DD-MM-YYYY
    r'(\d{2}[/\-]\d{2}[/\-]\d{4})',
    # DD/MM/YY
    r'(\d{2}[/\-]\d{2}[/\-]\d{2})',
    # Month name YYYY  e.g. JUN 2026 / JUNE 2026
    r'((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s.\-/]?\d{4})',
    # MM/YYYY  or  MM-YYYY
    r'(\d{1,2}[/\-]\d{4})',
    # MM/YY  or  MM-YY
    r'(\d{1,2}[/\-]\d{2})',
]


def _extract_date_from_text(text):
    """Find the first plausible date in a line of text."""
    text = _fix_ocr_errors(text)
    for pat in _DATE_PATTERNS:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            if _validate_date(candidate):
                return candidate
    return None


def _validate_date(date_str):
    """Return True if the date string looks like a plausible medicine expiry."""
    if not date_str:
        return False
    s = date_str.strip()
    try:
        # MM/YYYY or MM-YYYY
        if re.match(r'^\d{1,2}[/\-]\d{4}$', s):
            parts = re.split(r'[/\-]', s)
            return 1 <= int(parts[0]) <= 12 and 2020 <= int(parts[1]) <= 2050
        # MM/YY or MM-YY
        if re.match(r'^\d{1,2}[/\-]\d{2}$', s):
            parts = re.split(r'[/\-]', s)
            month = int(parts[0])
            year = 2000 + int(parts[1])
            return 1 <= month <= 12 and 2020 <= year <= 2050
        # DD/MM/YYYY
        if re.match(r'^\d{2}[/\-]\d{2}[/\-]\d{4}$', s):
            parts = re.split(r'[/\-]', s)
            return 1 <= int(parts[0]) <= 31 and 1 <= int(parts[1]) <= 12 \
                   and 2020 <= int(parts[2]) <= 2050
        # DD/MM/YY
        if re.match(r'^\d{2}[/\-]\d{2}[/\-]\d{2}$', s):
            parts = re.split(r'[/\-]', s)
            year = 2000 + int(parts[2])
            return 1 <= int(parts[0]) <= 31 and 1 <= int(parts[1]) <= 12 \
                   and 2020 <= year <= 2050
        # Month name YYYY
        if re.match(r'^[A-Za-z]{3}', s):
            return True
    except (ValueError, IndexError):
        pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# EXPIRY DATE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _extract_expiry_raw(lines, raw_ocr_lines=None, full_text=None):
    """
    Robust expiry extraction with:
      - Fuzzy EXP keyword matching
      - Same-line date search
      - Next-line date search (keyword on line N, date on line N+1)
      - Fallback: scan all lines for any valid date
      - Ignores MFG/Manufacturing dates
    """
    # ── Pass 1: look for EXP keyword and extract associated date ──────────
    for i, line in enumerate(lines):
        # Skip manufacturing date lines
        if re.search(r'\b(MFG|MFD|MANUFACTURED|MANUFACTURING)\b', line, re.I):
            continue

        tokens = re.split(r'[\s:.,|]+', line)
        is_exp = any(_is_exp_keyword(t) for t in tokens)

        if is_exp:
            # Same line
            date = _extract_date_from_text(line)
            if date:
                print(f"[EXPIRY] Same-line match: {date}")
                return date

            # Next line
            if i + 1 < len(lines):
                date = _extract_date_from_text(lines[i + 1])
                if date:
                    print(f"[EXPIRY] Next-line match: {date}")
                    return date

    # ── Pass 2: fallback – any valid date that is NOT on an MFG line ──────
    for line in lines:
        if re.search(r'\b(MFG|MFD|MANUFACTURED|MANUFACTURING)\b', line, re.I):
            continue
        date = _extract_date_from_text(line)
        if date:
            print(f"[EXPIRY] Fallback date: {date}")
            return date

    # Pass 3: Unified LLM fallback is handled in detect_pharma()
    # to avoid redundant API calls and keep extraction logic centralized.
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MEDICINE NAME EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

_SKIP_PATTERNS = [
    r'^(mg|ml|tablet|capsule|cap|tab|strip|pack|box|sachet)\s*$',
    r'^(exp|expiry|use.?before|mfd|mfg|batch|bn|lot|www|http)',
    r'^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}$',   # pure date
    r'^[A-Z0-9]{5,20}$',                        # batch numbers
    r'^\d+\s*(mg|ml|mcg|iu)$',                  # dose only
    r'^rs\.?\s*\d+',                             # price
    r'^\d{4,}$',                                 # long number
]


def _extract_medicine_name(lines):
    """
    Find the medicine name:
      - Usually in bold / ALL-CAPS / larger font → appears near top
      - Skip dosing lines, batch numbers, dates, prices
    """
    # Priority 1: pattern like PARACIP-500, DOLO-650, AMOX-500
    dosage_re = re.compile(r'\b([A-Za-z]{3,}(?:\s[A-Za-z]+)*[\s\-]\d+(?:mg|ml|mcg|iu)?)\b',
                           re.IGNORECASE)
    for line in lines[:15]:
        m = dosage_re.search(line)
        if m:
            name = re.sub(r'[^A-Za-z0-9\s\-\+\.]', '', m.group(1)).strip()
            if len(name) >= 4:
                return name

    # Priority 2: Fuzzy match against common medicines
    common_meds = ["CALPOL", "DOLO", "PARACETAMOL", "PANADOL", "AUGMENTIN", "CROCIN", 
                   "AZITHRAL", "LIMCEE", "ZIFI", "TAXIM", "TELMA", "AMLOKIND"]
    
    for line in lines[:10]:
        line_upper = line.upper()
        for med in common_meds:
            if med in line_upper:
                return med # Found an exact match within the line

    # Priority 3: first meaningful non-skipped line
    for line in lines[:10]:
        line = line.strip()
        if len(line) < 3:
            continue
        if any(re.match(p, line, re.IGNORECASE) for p in _SKIP_PATTERNS):
            continue
        cleaned = re.sub(r'[^A-Za-z0-9\s\-\+\.]', '', line).strip()
        if len(cleaned) >= 5 and re.search(r'[A-Za-z]{4,}', cleaned):
            return cleaned

    return None


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_image(image_input):
    if isinstance(image_input, bytes):
        arr = np.frombuffer(image_input, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, str):
        img = cv2.imread(image_input)
    else:
        raise ValueError("image_input must be bytes or a file path string")
    if img is None:
        raise ValueError("Failed to load image — check path or file format")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def detect_pharma(image_input):
    """
    Analyze a medicine label image and return:
      {
        "medicine_name": str | "Not Detected",
        "expiry_date":   str | "Not Detected",
        "full_text":     str,
        "status":        "success" | "failed",
        "confidence":    float,
      }
    """
    result = {
        "medicine_name": "Not Detected",
        "expiry_date": "Not Detected",
        "full_text": "",
        "status": "failed",
        "confidence": 0.0,
    }

    try:
        img = load_image(image_input)
    except Exception as e:
        print(f"[ERROR] load_image: {e}")
        return result

    # Preprocess and OCR
    variants = _preprocess_image(img)
    full_text = _ocr_all_variants(variants)
    full_text = _fix_ocr_errors(full_text)

    print(f"[DEBUG] OCR raw text:\n{full_text}\n{'-'*40}")

    lines = [l.strip() for l in full_text.splitlines() if l.strip()]

    # Extract fields (Rule-based first)
    med_name = _extract_medicine_name(lines)
    expiry = _extract_expiry_raw(lines, full_text=full_text)

    # ── LLM fallback for BOTH name and expiry ─────────────────────────────
    if not med_name or med_name == "Not Detected" or not expiry or expiry == "Not Detected":
        try:
            print("[DEBUG] Rule-based extraction incomplete. Trying Unified AI fallback (Gemini/Claude)...")
            from services.advanced_text_extraction import UnifiedAIExtractor
            llm_res = UnifiedAIExtractor.extract_details(full_text)
            
            if not med_name or med_name == "Not Detected":
                if llm_res.get("medicine_name"):
                    print(f"[DEBUG] AI found name: {llm_res['medicine_name']}")
                    med_name = llm_res["medicine_name"]
            
            if not expiry or expiry == "Not Detected":
                if llm_res.get("expiry_date"):
                    print(f"[DEBUG] AI found expiry: {llm_res['expiry_date']}")
                    expiry = llm_res["expiry_date"]
        except Exception as e:
            print(f"[DEBUG] AI fallback error: {e}")

    # ---------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------

    result["medicine_name"] = med_name or "Not Detected"
    result["expiry_date"] = expiry or "Not Detected"
    result["full_text"] = full_text

    if med_name or expiry:
        result["status"] = "success"
        result["confidence"] = 0.85 if (med_name and expiry) else 0.50

    print(f"[RESULT] {json.dumps({k: result[k] for k in ('medicine_name','expiry_date','status')}, indent=2)}")
    return result
