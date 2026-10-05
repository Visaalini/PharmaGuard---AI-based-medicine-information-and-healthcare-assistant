"""
Enhanced OCR-based extraction for medicine packaging
Improves batch number and expiry date detection from medicine package images
"""

import cv2
import numpy as np
import re
import os
from datetime import datetime, date

# ============================================================================
# IMAGE PREPROCESSING - Multiple strategies for better OCR
# ============================================================================

def preprocess_for_ocr(image_path, strategy='auto'):
    """
    Preprocess image for optimal OCR results
    
    Strategies:
    - 'auto': Try multiple and return best
    - 'grayscale': Standard grayscale conversion
    - 'threshold': Binary threshold
    - 'adaptive': Adaptive threshold for uneven lighting
    - 'clahe': Contrast enhancement
    - 'denoise': Remove noise while preserving text
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, "Image could not be loaded"
    
    h, w = img.shape[:2]
    
    # Resize small images for better OCR (minimum 1500px width)
    scale = 1500 / w if w < 1500 else 1
    if scale > 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    strategies = {
        'grayscale': gray,
        'threshold': apply_threshold(gray),
        'adaptive': apply_adaptive_threshold(gray),
        'clahe': apply_clahe(gray),
        'denoise': apply_denoise(gray),
        'inverted': apply_inverted(gray),
    }
    
    if strategy == 'auto':
        return strategies, "Multiple strategies ready"
    elif strategy in strategies:
        return strategies[strategy], f"Strategy: {strategy}"
    else:
        return gray, "Default grayscale"


def apply_threshold(gray):
    """Binary threshold"""
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def apply_adaptive_threshold(gray):
    """Adaptive threshold for uneven lighting"""
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return adaptive


def apply_clahe(gray):
    """CLAHE for contrast enhancement"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def apply_denoise(gray):
    """Bilateral filter - removes noise while preserving edges"""
    return cv2.bilateralFilter(gray, 9, 75, 75)


def apply_inverted(gray):
    """Inverted binary - white text on dark background"""
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_not(thresh)


# ============================================================================
# TESSERACT CONFIGURATIONS
# ============================================================================

# Multiple PSM modes for different text layouts
TESSERACT_CONFIGS = {
    'auto': '--oem 3 --psm 3',           # Fully automatic
    'block': '--oem 3 --psm 6',          # Uniform block of text
    'sparse': '--oem 3 --psm 11',        # Sparse text (good for scattered labels)
    'single': '--oem 3 --psm 4',         # Single line of text
    'single_vert': '--oem 3 --psm 5',     # Single vertical line
    'dense': '--oem 3 --psm 2',          # Dense uniform text
}

# Whitelist for medicine labels (common characters)
CHAR_WHITELIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz/-.: '


# ============================================================================
# REGEX PATTERNS FOR BATCH NUMBER
# ============================================================================

BATCH_PATTERNS = [
    # Priority 1: Explicit keywords (most reliable)
    (r'BATCH\s*(?:NO\.?|NUMBER|#|CODE)?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'BN_KEYWORD'),
    (r'B\.?NO\.?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'B_NO'),
    (r'BN[\s]*NO\.?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'BN'),
    (r'LOT[\s]*(?:NO\.?|NUMBER|#)?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'LOT'),
    (r'L/?NO\.?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'L_NO'),
    (r'SERIAL[\s]*(?:NO\.?)?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'SERIAL'),
    (r'MFG[\s]*(?:NO\.?)?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'MFG_NO'),
    (r'MFD[\s]*(?:NO\.?)?[\s:.\-]*([A-Z0-9][A-Z0-9/_\-\.]{2,25})', 'MFD_NO'),
    
    # Priority 2: Letter + Number combinations (alphanumeric)
    (r'\b([A-Z]{1,4}\d{4,12}[A-Z0-9]{0,5})\b', 'ALPHANUM_1'),
    (r'\b([A-Z]{2,4}[-/]?\d{3,12})\b', 'ALPHANUM_2'),
    (r'\b([A-Z]\d{5,15})\b', 'ALPHANUM_3'),
    (r'\b(\d{3,5}[A-Z]{2,5}\d{2,8})\b', 'ALPHANUM_4'),
    
    # Priority 3: Aggressive patterns (less strict)
    (r'\bBATCH[\s]*([A-Z0-9]{3,20})\b', 'BATCH_NUM'),
    (r'\bBN[\s]*([A-Z0-9]{3,20})\b', 'BN_NUM'),
    (r'\bLOT[\s]*([A-Z0-9]{3,20})\b', 'LOT_NUM'),
]

# Words to exclude from batch number results
BATCH_EXCLUDE = {
    'THE', 'AND', 'FOR', 'NOT', 'ALL', 'ARE', 'BUT', 'CAN', 'HAS', 'HAD',
    'YEAR', 'DATE', 'TIME', 'BEST', 'WORST', 'THIS', 'THAT', 'WITH',
    'FROM', 'NAME', 'FOOD', 'GOOD', 'LOOK', 'MORE', 'SOME', 'WILL',
}


# ============================================================================
# REGEX PATTERNS FOR EXPIRY DATE
# ============================================================================

EXPIRY_PATTERNS = [
    # Priority 1: Explicit keywords (most reliable)
    (r'EXP(?:IRY|IRED?|\.)?[\s:.\-]*DATE[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'EXP_DATE_KEYWORD'),
    (r'EXP(?:IRY|IRED?|\.)?[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'EXP_KEYWORD'),
    (r'EXP(?:IRY|IRED?|\.)?[\s:.\-]*(\d{1,2}[\s.\-/]\d{2})', 'EXP_MM_YY'),
    (r'EXP(?:IRY|IRED?|\.)?[\s:.\-]*(\d{1,2}[\s.\-/]\d{4})', 'EXP_MM_YYYY'),
    (r'EXPIRES?[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'EXPIRES'),
    (r'USE[\s]+BEFORE[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'USE_BEFORE'),
    (r'BEST[\s]+BEFORE[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'BEST_BEFORE'),
    (r'VALID[\s]+(?:TILL|UNTIL|UPTO|THRU)[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'VALID_TILL'),
    (r'MFG[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'MFG_DATE'),
    (r'MFD[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 'MFD_DATE'),
    
    # Priority 2: Month name formats
    (r'EXP[^\n]{0,15}((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?\d{2,4})', 'EXP_MONTH_NAME'),
    (r'((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?\d{4})', 'MONTH_NAME'),
    (r'(?:EXP|EXPIRY)[\s:.\-]*([A-Z]{3}[\s.\-/]?\d{2})', 'EXP_SHORT_MONTH'),
    
    # Priority 3: Date patterns without keywords
    (r'\b(\d{1,2}[/-]\d{2})\b', 'MM_YY'),           # MM/YY
    (r'\b(\d{1,2}[/-]\d{4})\b', 'MM_YYYY'),        # MM/YYYY
    (r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', 'DD_MM_YY'),
    (r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b', 'YYYY_MM_DD'),
]

# Month abbreviation mapping
MONTH_MAP = {
    'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
    'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
}


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

def extract_batch_number(text):
    """Extract batch number from OCR text"""
    if not text or not isinstance(text, str):
        return None, 0.0
    
    text = text.strip().upper()
    if len(text) < 3:
        return None, 0.0
    
    # Try patterns in order of priority
    for pattern, pattern_type in BATCH_PATTERNS:
        match = re.search(pattern, text)
        if match:
            val = match.group(1).strip().rstrip('/-.')
            
            # Clean and validate
            val = re.sub(r'[^A-Z0-9]', '', val)
            
            # Filter false positives
            if len(val) >= 3 and val not in BATCH_EXCLUDE:
                # Check it's not just a year
                if not (len(val) == 4 and val.isdigit() and 2000 <= int(val) <= 2050):
                    confidence = get_batch_confidence(val, pattern_type)
                    return val, confidence
    
    return None, 0.0


def get_batch_confidence(value, pattern_type):
    """Calculate confidence based on pattern type and value characteristics"""
    confidence = 0.5  # Base confidence
    
    # Higher confidence for explicit keyword patterns
    if 'KEYWORD' in pattern_type or pattern_type in ['BN', 'B_NO', 'LOT', 'SERIAL']:
        confidence = 0.85
    
    # Higher confidence for longer alphanumeric values
    if len(value) >= 8:
        confidence += 0.1
    
    # Check for common batch format indicators
    if '/' in value or '-' in value:
        confidence += 0.05
    
    return min(confidence, 0.95)


def extract_expiry_date(text):
    """Extract expiry date from OCR text"""
    if not text or not isinstance(text, str):
        return None, 0.0
    
    text = text.strip().upper()
    if len(text) < 3:
        return None, 0.0
    
    # Try patterns in order of priority
    for pattern, pattern_type in EXPIRY_PATTERNS:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1).strip()
            
            # Parse and validate the date
            parsed_date, confidence = parse_expiry_date(date_str, pattern_type)
            if parsed_date:
                return parsed_date, confidence
    
    return None, 0.0


def parse_expiry_date(date_str, pattern_type):
    """Parse and validate expiry date string"""
    if not date_str:
        return None, 0.0
    
    date_str = date_str.strip().upper()
    now = date.today()
    
    try:
        # Try various date formats
        formats = [
            '%m/%y', '%m/%Y', '%m-%y', '%m-%Y',
            '%d/%m/%y', '%d/%m/%Y', '%d-%m-%y', '%d-%m-%Y',
            '%d/%m/%Y', '%Y-%m-%d',
            '%b %Y', '%b %y',  # JAN 2025, JAN 25
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt).date()
                
                # Validate reasonable date range
                if 2020 <= parsed.year <= 2050:
                    # Calculate confidence
                    confidence = 0.9
                    diff_months = (parsed.year - now.year) * 12 + (parsed.month - now.month)
                    
                    if diff_months < 0:
                        confidence -= 0.1  # Expired
                    elif diff_months <= 3:
                        confidence -= 0.05  # Expiring soon
                    
                    return parsed.strftime('%m/%Y'), min(confidence, 0.95)
            except ValueError:
                continue
        
        # Try month abbreviation
        for month_abbr, month_num in MONTH_MAP.items():
            if month_abbr in date_str:
                # Extract year
                year_match = re.search(r'\d{2,4}', date_str)
                if year_match:
                    year = int(year_match.group())
                    year = 2000 + year if year <= 40 else year
                    
                    if 2020 <= year <= 2050:
                        return f"{month_num:02d}/{year}", 0.85
        
    except Exception:
        pass
    
    return None, 0.0


def validate_expiry_status(expiry_str):
    """Returns: EXPIRED, EXPIRING SOON, VALID, or None"""
    if not expiry_str:
        return None
    
    now = date.today()
    
    try:
        # Try to parse the date
        for fmt in ['%m/%Y', '%m/%y', '%d/%m/%Y', '%d/%m/%y']:
            try:
                exp_date = datetime.strptime(expiry_str, fmt).date()
                diff_months = (exp_date.year - now.year) * 12 + (exp_date.month - now.month)
                
                if diff_months < 0:
                    return "EXPIRED"
                elif diff_months <= 3:
                    return "EXPIRING SOON"
                else:
                    return "VALID"
            except ValueError:
                continue
    except Exception:
        pass
    
    return None


# ============================================================================
# MAIN EXTRACTION FUNCTION
# ============================================================================

def extract_batch_and_expiry(image_path):
    """
    Main function to extract batch number and expiry date from medicine image
    
    Returns:
        dict: {
            'batch_number': str or None,
            'batch_confidence': float,
            'expiry_date': str or None,
            'expiry_confidence': float,
            'expiry_status': str or None,
            'raw_text': str,
        }
    """
    import pytesseract
    
    # Set Tesseract path
    if os.name == 'nt':  # Windows
        tesseract_path = os.environ.get('TESSERACT_CMD', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    # Preprocess image
    preprocess_results, _ = preprocess_for_ocr(image_path, strategy='auto')
    
    if not preprocess_results:
        return {'error': 'Could not load image'}
    
    # Try OCR with multiple strategies
    best_text = ""
    
    for strategy_name, processed_img in preprocess_results.items():
        for config_name, config in TESSERACT_CONFIGS.items():
            try:
                # Add whitelist config
                whitelist_config = config + f' -c tessedit_char_whitelist={CHAR_WHITELIST}'
                
                text = pytesseract.image_to_string(processed_img, config=whitelist_config)
                text = text.strip()
                
                # Keep the longest valid result
                if len(text) > len(best_text):
                    best_text = text
                    
            except Exception:
                continue
    
    if not best_text:
        best_text = "No text detected"
    
    # Extract fields
    batch_number, batch_conf = extract_batch_number(best_text)
    expiry_date, expiry_conf = extract_expiry_date(best_text)
    expiry_status = validate_expiry_status(expiry_date) if expiry_date else None
    
    return {
        'batch_number': batch_number,
        'batch_confidence': batch_conf,
        'expiry_date': expiry_date,
        'expiry_confidence': expiry_conf,
        'expiry_status': expiry_status,
        'raw_text': best_text,
    }


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_extraction.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    print("=" * 60)
    print(f"Extracting from: {image_path}")
    print("=" * 60)
    
    result = extract_batch_and_expiry(image_path)
    
    print("\nRESULTS:")
    print("-" * 60)
    print(f"Batch Number:     {result.get('batch_number', 'NOT DETECTED')}")
    print(f"  Confidence:   {result.get('batch_confidence', 0):.1%}")
    print(f"\nExpiry Date:      {result.get('expiry_date', 'NOT DETECTED')}")
    print(f"  Confidence:   {result.get('expiry_confidence', 0):.1%}")
    print(f"  Status:        {result.get('expiry_status', 'N/A')}")
    
    print("\nRaw Text:")
    print("-" * 60)
    print(result.get('raw_text', '')[:500])