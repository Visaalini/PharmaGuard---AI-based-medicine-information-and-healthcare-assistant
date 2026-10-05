from models.ocr_reader import read_text
from services.advanced_text_extraction import UnifiedAIExtractor, FieldExtractor
import cv2
import re
import os
from concurrent.futures import ThreadPoolExecutor

_OCR_EMPTY = {"no text detected", "image could not be loaded", "ocr unavailable",
               "ocr error", "tesseract not found"}

def _valid_ocr(text):
    if not text:
        return None
    for skip in _OCR_EMPTY:
        if text.lower().startswith(skip):
            return None
    return text

def detect_barcode(image_path):
    try:
        try:
            from pyzbar.pyzbar import decode as decode_barcode
        except Exception:
            return None

        img = cv2.imread(image_path)
        if img is None:
            return None
        
        barcodes = decode_barcode(img)
        if barcodes:
            return barcodes[0].data.decode("utf-8")
        return None
    except Exception as e:
        print(f"Barcode detection error: {e}")
        return None

def extract_expiry_date(text):
    """Extract expiry date using regex first, then LLM fallback."""
    if not _valid_ocr(text):
        return "Unable to read", None

    print(f"[DEBUG] Extracting expiry date from: {text[:300]}...")

    # Primary: regex-based extraction (faster, works offline)
    exp_date, exp_conf = FieldExtractor.extract_expiry_date(text)
    if exp_date:
        status = FieldExtractor.check_expiry_status(exp_date)
        print(f"[REGEX] Expiry: {exp_date}, Status: {status}")
        return exp_date, status

    # Fallback: LLM-based extraction via UnifiedAIExtractor (Gemini)
    try:
        llm_result = UnifiedAIExtractor.extract_details(text)
        if llm_result.get("expiry_date"):
            status = FieldExtractor.check_expiry_status(llm_result["expiry_date"])
            print(f"[AI] Expiry: {llm_result['expiry_date']}, Status: {status}")
            return llm_result["expiry_date"], status
    except Exception as e:
        print(f"[AI] LLM extraction failed: {e}")

    return "Unable to read", None

def extract_medicine_name(text):
    text = _valid_ocr(text)
    if not text:
        return "Unable to read"

    print(f"[DEBUG] Extracting medicine name from: {text[:500]}...")

    lines = text.strip().split('\n')

    # Fix 2c: Search for dosage pattern first (PARACIP-500, DOLO-650)
    dosage_pattern = re.compile(r'\b([A-Za-z]+(?:-\d+)+)\b')
    for line in lines:
        match = dosage_pattern.search(line)
        if match:
            result = match.group(1)
            print(f"[DEBUG] Found medicine name (dosage pattern): {result}")
            return result

    # Search for common medicine patterns (e.g., ALAXAN FR, IBUPROFEN PARACETAMOL)
    # Pattern: ALL CAPS words that could be medicine names
    med_pattern = re.compile(r'\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b')
    for line in lines[:15]:  # Check more lines
        matches = med_pattern.findall(line)
        for match in matches:
            # Skip common non-medicine words
            skip = ['EXP', 'MFG', 'MFD', 'BATCH', 'LOT', 'USE', 'BEFORE', 'BEST',
                    'VALID', 'UNTIL', 'TILL', 'MG', 'ML', 'TABLET', 'CAPSULE']
            if match not in skip and len(match) >= 3:
                print(f"[DEBUG] Found medicine name (caps pattern): {match}")
                return match

    skip_words = re.compile(
        r'^(MG|ML|TABLET|CAPSULE|STRIP|PACK|BOX|EXP|MFD|MFG|BATCH|BN|LOT|WWW|HTTP|'
        r'MANUFACTURED|MARKETED|DISTRIBUTED|KEEP|WARNING|CAUTION|SALE|INGREDIENT|'
        r'DOSE|DOSAGE|USAGE|USE|DIRECTION|EACH|CONTAINS|COMPOSITION|'
        r'SCHEDULE|DRUG|PHARMACIST|PRESCRIPTION|SIDE|EFFECT|'
        r'FOLIC|ACID|MOREOVER|BETAPHARM|SOOMG|PARACETAMOL|500MG|250MG|'
        r'\d+MG|\d+ML|STRIP|PACK|BLISTER|\d)',
        re.IGNORECASE
    )

    for line in lines[:15]:  # Check more lines
        line = line.strip()
        if len(line) >= 2:
            cleaned = re.sub(r'[^A-Za-z0-9\s\-\+\.]', '', line).strip()
            if len(cleaned) >= 3 and not skip_words.match(cleaned):
                if re.search(r'[A-Za-z]{2,}', cleaned):
                    print(f"[DEBUG] Found medicine name: {cleaned}")
                    return cleaned

    patterns = [
        r'(?:MEDICINE|DRUG|PRODUCT|BRAND|NAME)[\s:.\-]*([A-Za-z][A-Za-z0-9\s\-]{2,40})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text.upper())
        if match:
            result = match.group(1).strip()
            if len(result) >= 3:
                print(f"[DEBUG] Found medicine via pattern: {result}")
                return result

    # Last resort: return first line with decent text
    for line in lines:
        cleaned = re.sub(r'[^A-Za-z0-9\s\-\+\.]', '', line.strip()).strip()
        if len(cleaned) >= 3 and re.search(r'[A-Za-z]{2,}', cleaned):
            print(f"[DEBUG] Last resort: {cleaned}")
            return cleaned

    return "Unable to read"


def detect_medicine(image_path, user_name=None, user_phone=None):
    from models.predict import run_cnn
    from services.advanced_text_extraction import AdvancedTextExtractor
    
    print(f"[TURBO] Starting parallel analysis for: {image_path}")
    
    # Pre-resize image for faster processing across all modules
    # 800px is enough for OCR and saves massive CPU time
    img = cv2.imread(image_path)
    if img is not None:
        h, w = img.shape[:2]
        if w > 800:
            scale = 800.0 / w
            img = cv2.resize(img, (800, int(h * scale)))
            cv2.imwrite(image_path, img) # Overwrite with optimized version
    
    # Use Parallel Execution for independent tasks
    with ThreadPoolExecutor(max_workers=3) as executor:
        cnn_task = executor.submit(run_cnn, image_path)
        ocr_task = executor.submit(lambda: AdvancedTextExtractor().extract_text_from_image(image_path))
        barcode_task = executor.submit(detect_barcode, image_path)
        
        # Wait for all (this takes the time of the SLOWEST task, not the SUM)
        prediction, confidence = cnn_task.result()
        text = ocr_task.result()
        barcode = barcode_task.result()

    if prediction is None:
        return {
            "prediction":    "detection_failed",
            "confidence":    None,
            "text":          "No text detected",
            "barcode":       None,
            "expiry_date":   None,
            "expiry_status": None,
            "medicine_name": None
        }

    # Extract fields from the retrieved text
    medicine_name = extract_medicine_name(text)
    expiry_date, expiry_status = extract_expiry_date(text)

    if prediction == "fake":
        return {
            "prediction":    "fake",
            "confidence":    confidence,
            "text":          text,
            "barcode":       barcode,
            "expiry_date":   None,
            "expiry_status": None,
            "medicine_name": None
        }

    return {
        "prediction":    prediction,
        "confidence":    confidence,
        "text":          text,
        "barcode":       barcode,
        "expiry_date":   expiry_date,
        "expiry_status": expiry_status,
        "medicine_name": medicine_name
    }
