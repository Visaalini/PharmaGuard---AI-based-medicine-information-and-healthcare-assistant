"""
Advanced Text Extraction & Field Parsing
- Uses EasyOCR + Tesseract for redundancy
- Extracts: Medicine Name, Expiry Date
- Handles various text formats and layouts
- Confidence scoring for each extraction
- LLM-based expiry extraction (Claude) for complex labels
"""

import re
import json
import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, date
import os
import requests
import base64

# Try importing both OCR engines
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


class AdvancedTextExtractor:
    """Advanced text extraction using multiple OCR engines.

    Uses the shared EasyOCR READER from detector.py (loaded once at startup)
    to avoid 30-60s per-request initialization.
    """

    _shared_reader = None

    def __init__(self, use_easyocr: bool = True, use_tesseract: bool = True):
        self.use_easyocr = use_easyocr and HAS_EASYOCR
        self.use_tesseract = use_tesseract and HAS_TESSERACT
        self.extraction_log = []

    @classmethod
    def _get_reader(cls):
        if cls._shared_reader is None:
            try:
                from detector import READER
                cls._shared_reader = READER
                print("[INFO] Using shared EasyOCR reader from detector")
            except (ImportError, AttributeError) as e:
                print(f"[WARNING] Could not load shared EasyOCR reader: {e}")
                try:
                    cls._shared_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                    print("[INFO] Fallback: loaded EasyOCR reader directly")
                except Exception as e2:
                    print(f"[ERROR] EasyOCR init failed: {e2}")
                    cls._shared_reader = None
        return cls._shared_reader

    def extract_text_from_image(self, image_path: str, preprocessed_variant: Optional[np.ndarray] = None) -> str:
        """
        Extract text from image using multiple OCR engines

        Priority:
        1. EasyOCR with multiple preprocessed variants (best for varied text)
        2. Tesseract with multiple configs
        3. Fallback to empty string

        Returns: Combined extracted text
        """
        if preprocessed_variant is None:
            img = cv2.imread(image_path)
            if img is None:
                print(f"[ERROR] Cannot read image at {image_path}")
                return "No text detected"
            print(f"[DEBUG] Image loaded: shape={img.shape}, dtype={img.dtype}")
        else:
            img = preprocessed_variant
            print(f"[DEBUG] Using preprocessed image: shape={img.shape}")

        texts = []

        # Note: EasyOCR is DISABLED for speed.
        # if self.use_easyocr:
        #    ... (code removed for speed)

        # Try Tesseract as fallback with multiple PSM modes
        if self.use_tesseract:
            try:
                print("[DEBUG] Attempting Tesseract extraction...")
                tess_text = self._extract_with_tesseract(img)
                if tess_text and len(tess_text.strip()) > 0:
                    texts.append(("Tesseract", tess_text))
                    print(f"[DEBUG] Tesseract found {len(tess_text.split())} words")
            except Exception as e:
                print(f"[DEBUG] Tesseract failed: {e}")

        # Combine results
        if texts:
            # If only one engine succeeded, use that
            if len(texts) == 1:
                combined = texts[0][1]
                print(f"[DEBUG] Using {texts[0][0]} results")
            else:
                # If both succeeded, use the one with more text
                if len(texts[0][1]) >= len(texts[1][1]):
                    combined = texts[0][1]
                    print(f"[DEBUG] Using {texts[0][0]} results (more text)")
                else:
                    combined = texts[1][1]
                    print(f"[DEBUG] Using {texts[1][0]} results (more text)")

            self.extraction_log.append({
                'method': texts[0][0],
                'text_length': len(combined),
                'timestamp': datetime.now().isoformat()
            })

            return combined
        else:
            return "No text detected"

    @staticmethod
    def _preprocess_image(img):
        """Create multiple preprocessed variants for OCR."""
        results = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        if w < 1500:
            scale = 1500.0 / w
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        denoised = cv2.bilateralFilter(contrast, 11, 95, 95)

        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        results.append(contrast)
        results.append(adaptive)

        return results

    def _extract_with_easyocr(self, image_input, reader) -> str:
        """Extract text using EasyOCR on multiple preprocessed variants."""
        if reader is None:
            return ""

        try:
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    return ""
            else:
                img = image_input.copy()

            # Add original image as a variant (color)
            variants = [img]
            variants.extend(self._preprocess_image(img))

            best_text = ""
            for idx, arr in enumerate(variants):
                # Convert to RGB for EasyOCR
                if len(arr.shape) == 2:  # Grayscale
                    arr_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                else:  # Color
                    arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

                # Resize if too small (OCR works better on larger images)
                h, w = arr_rgb.shape[:2]
                if w < 1000:
                    scale = 1000.0 / w
                    arr_rgb = cv2.resize(arr_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                results = reader.readtext(
                    arr_rgb,
                    contrast_ths=0.3,  # Lower = more sensitive to low contrast
                    text_threshold=0.4,  # Lower = detect more text
                    link_threshold=0.25,  # Lower = more aggressive text linking
                    width_ths=0.7,  # Higher = allow wider text boxes
                    height_ths=0.7,  # Higher = allow taller text boxes
                )

                texts = []
                for detection in results:
                    text = detection[1]
                    confidence = detection[2]
                    if confidence > 0.2 and text.strip():  # Lower threshold to catch more text
                        texts.append(text.strip())

                candidate = "\n".join(texts)
                print(f"[DEBUG] Variant {idx}: found {len(texts)} text items, total {len(candidate)} chars")

                if len(candidate) > len(best_text):
                    best_text = candidate
                
                # EARLY EXIT: If we found enough text with EasyOCR, don't try other variants
                if len(candidate.strip()) > 100 and "EXP" in candidate.upper():
                    print(f"[INFO] High-quality text found on variant {idx}. Exiting EasyOCR early.")
                    return candidate

            return best_text
        except Exception as e:
            print(f"[ERROR] EasyOCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _extract_with_tesseract(self, image_input) -> str:
        """Extract text using Tesseract with multiple configurations"""
        if not self.use_tesseract or not HAS_TESSERACT:
            return ""

        try:
            # Set tesseract path if needed
            tesseract_path = self._find_tesseract()
            if tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = tesseract_path

            # Try multiple PSM (Page Segmentation Mode) configurations
            configs = [
                "--oem 3 --psm 6",   # Uniform block of text
                "--oem 3 --psm 11",  # Sparse text
                "--oem 3 --psm 3",   # Fully automatic
            ]

            best_text = ""
            for config in configs:
                try:
                    # Handle both file path and numpy array
                    if isinstance(image_input, str):
                        text = pytesseract.image_to_string(image_input, config=config)
                    else:
                        text = pytesseract.image_to_string(image_input, config=config)

                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text

                except Exception as e:
                    print(f"[DEBUG] Tesseract config '{config}' failed: {e}")
                    continue

            return best_text
        except Exception as e:
            print(f"[ERROR] Tesseract extraction failed: {e}")
            return ""

    @staticmethod
    def _find_tesseract() -> Optional[str]:
        """Find tesseract installation path"""
        env_path = os.environ.get("TESSERACT_CMD", "")
        if env_path and os.path.exists(env_path):
            return env_path

        win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in win_paths:
            if os.path.isfile(path):
                return path

        return None


class FieldExtractor:
    """
    Extract specific fields from OCR text:
    - Medicine Name
    - Expiry Date
    """

    CONFIDENCE_THRESHOLDS = {
        'medicine_name': 0.5,
        'expiry_date': 0.7,
    }

    @staticmethod
    def extract_batch_number(text: str) -> Tuple[Optional[str], float]:
        """Extract batch number from OCR text"""
        if not text or len(text.strip()) == 0:
            return None, 0.0

        text_upper = text.upper()
        patterns = [
            (r'BATCH\s*(?:NO\.?|NUMBER|#|CODE)?\s*[:=.\-]?\s*([A-Z0-9][A-Z0-9/_\-\.]{2,30})', 0.98),
            (r'B\.?[\s]*NO\.?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.90),
            (r'BN\.?\s*(?:NO\.?)?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.93),
            (r'LOT\s*(?:NO\.?|NUMBER|#)?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.88),
            (r'L\s*NO\.?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.85),
            (r'\b([A-Z]{2,4}\d{6,12})\b', 0.70),
            (r'\b([A-Z]\d{7,12})\b', 0.65),
        ]
        for pattern, base_conf in patterns:
            m = re.search(pattern, text_upper)
            if m:
                val = m.group(1).strip().rstrip('/-.')
                if len(val) >= 3 and not re.match(r'^(19|20)\d{2}$', val):
                    return val, min(base_conf, 1.0)
        return None, 0.0

    @staticmethod
    def extract_medicine_name(text: str) -> Tuple[Optional[str], float]:
        """
        Extract medicine name from text

        Returns: (medicine_name, confidence)
        """
        if not text or len(text.strip()) == 0:
            return None, 0.0

        text_lines = text.strip().split('\n')

        # Skip common metadata lines
        skip_patterns = [
            r'^\s*(mg|ml|tablet|capsule|strip|pack|box)\s*$',
            r'^\s*(exp|mfg|batch|bn|lot|www|http)',
            r'^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # Dates
            r'^[A-Z0-9]{5,20}\s*$',  # Batch-like codes
        ]

        for line in text_lines[:8]:  # Check first 8 lines
            line = line.strip()

            # Skip if line matches skip patterns
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue

            # Look for alpha-numeric names (usually medicine names)
            if 3 <= len(line) <= 80 and re.search(r'[A-Za-z]{3,}', line):
                # Remove noise characters
                cleaned = re.sub(r'[^A-Za-z0-9\s\-\+\.]', '', line).strip()

                if len(cleaned) >= 3:
                    # Confidence based on line characteristics
                    confidence = 0.7
                    if len(cleaned) < 10:
                        confidence -= 0.1  # Too short
                    if cleaned[0].isupper():
                        confidence += 0.1  # Starts with uppercase

                    return cleaned, min(confidence, 1.0)

        return None, 0.0

    @staticmethod
    def extract_expiry_date(text: str) -> Tuple[Optional[str], float]:
        """
        Extract expiry date from text

        Handles formats:
        - EXP: 05/2026
        - Exp Date: 12/2025
        - Use Before: 31/12/2025
        - JAN 2026

        Returns: (expiry_date, confidence)
        """
        if not text or len(text.strip()) == 0:
            return None, 0.0

        text_upper = text.upper()

        # Keyword-based patterns (highest priority)
        keyword_patterns = [
            (r'EXP(?:IRY|IRED?|\.)?[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.98),
            (r'EXP(?:IRY|IRED?|\.)?[\s:.\-]*(\d{1,2}[-/.]\d{2,4})', 0.95),  # MM/YY
            (r'USE[\s]+BEFORE[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.90),
            (r'BEST[\s]+BEFORE[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.90),
            (r'VALID[\s]+(?:TILL|UNTIL|UPTO)[\s:.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.85),
            (r'EXP[^\n]{0,15}((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?\d{2,4})', 0.90),
        ]

        for pattern, confidence in keyword_patterns:
            match = re.search(pattern, text_upper)
            if match:
                date_str = match.group(1).strip()

                # Validate date format
                if FieldExtractor._validate_date_format(date_str):
                    return date_str, confidence

        # Fallback: any date-like pattern
        fallback_patterns = [
            (r'\b(\d{1,2}[/-]\d{2})\b', 0.65),  # MM/YY
            (r'\b(\d{1,2}[/-]\d{4})\b', 0.70),  # MM/YYYY
            (r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', 0.60),  # DD/MM/YYYY
        ]

        for pattern, confidence in fallback_patterns:
            matches = re.findall(pattern, text_upper)
            for date_str in matches:
                if FieldExtractor._validate_date_format(date_str):
                    return date_str, confidence

        return None, 0.0

    @staticmethod
    def _validate_date_format(date_str: str) -> bool:
        """Validate if string looks like a valid date"""
        date_str = date_str.strip()

        # MM/YY format
        if re.match(r'^\d{1,2}[/-]\d{2}$', date_str):
            parts = re.split(r'[-/]', date_str)
            try:
                month = int(parts[0])
                year = int(parts[1])
                # Valid month?
                if 1 <= month <= 12:
                    # Valid year (2020-2050)?
                    full_year = 2000 + year if year <= 40 else 1900 + year
                    if 2020 <= full_year <= 2050:
                        return True
            except ValueError:
                pass

        # MM/YYYY format
        if re.match(r'^\d{1,2}[/-]\d{4}$', date_str):
            parts = re.split(r'[-/]', date_str)
            try:
                month = int(parts[0])
                year = int(parts[1])
                if 1 <= month <= 12 and 2020 <= year <= 2050:
                    return True
            except ValueError:
                pass

        # DD/MM/YYYY format
        if re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$', date_str):
            return True

        # Month name format (JAN 2026)
        if re.match(r'^[A-Z]{3}\s?[/-]?\d{2,4}$', date_str):
            return True

        return False

    @staticmethod
    def check_expiry_status(date_str: Optional[str]) -> Optional[str]:
        """
        Check if medicine is expired

        Returns: 'EXPIRED', 'EXPIRING_SOON', 'VALID', or None
        """
        if not date_str:
            return None

        try:
            today = date.today()

            # Parse MM/YY format
            m = re.match(r'^(\d{1,2})[/-](\d{2})$', date_str.strip())
            if m:
                month = int(m.group(1))
                year = int(m.group(2))
                year = 2000 + year if year <= 40 else 1900 + year
                if 1 <= month <= 12:
                    exp_date = date(year, month, 28)  # Use day 28 for month-end
                    if exp_date < today:
                        return "EXPIRED"
                    if (exp_date.year - today.year) * 12 + (exp_date.month - today.month) <= 3:
                        return "EXPIRING_SOON"
                    return "VALID"

            # Parse MM/YYYY format
            m = re.match(r'^(\d{1,2})[/-](\d{4})$', date_str.strip())
            if m:
                month = int(m.group(1))
                year = int(m.group(2))
                if 1 <= month <= 12 and 2020 <= year <= 2050:
                    exp_date = date(year, month, 28)
                    if exp_date < today:
                        return "EXPIRED"
                    if (exp_date.year - today.year) * 12 + (exp_date.month - today.month) <= 3:
                        return "EXPIRING_SOON"
                    return "VALID"

            # Parse month name format
            m = re.match(r'^([A-Z]{3})\s?[/-]?(\d{2,4})$', date_str.strip().upper())
            if m:
                month_names = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                              "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
                month = month_names.get(m.group(1))
                if month:
                    year = int(m.group(2))
                    if len(m.group(2)) == 2:
                        year = 2000 + year if year <= 40 else 1900 + year
                    exp_date = date(year, month, 28)
                    if exp_date < today:
                        return "EXPIRED"
                    if (exp_date.year - today.year) * 12 + (exp_date.month - today.month) <= 3:
                        return "EXPIRING_SOON"
                    return "VALID"

        except Exception as e:
            print(f"[WARNING] Date validation error: {e}")

        return None




class GoogleGeminiExtractor:
    """FREE LLM-based extraction using Google Gemini via REST API to avoid Protobuf conflicts"""

    @staticmethod
    def extract_fields(ocr_text: str, image_path: str = None) -> dict:
        """Extract medicine name and expiry using Gemini"""
        import requests
        import base64
        import json
        
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key or "PASTE_YOUR_GEMINI_KEY_HERE" in api_key or api_key == "YOUR_GEMINI_API_KEY":
            return {"medicine_name": None, "expiry_date": None}

        prompt = f"""
        You are an expert Pharmaceutical Analyst. Extract critical information from a medicine package.

        **CRITICAL**: Look at the IMAGE carefully first. The image contains the actual medicine package.

        Your task: Find and extract:
        1. **MEDICINE NAME (Brand + Dosage)** - Usually the LARGEST text on the package
           Examples: "DERIPHYLLIN-100", "CALPOL-500MG", "PARACIP-650"
        
        2. **EXPIRY DATE** - Search for keywords: EXP, EXPIRY, VALID UNTIL, USE BEFORE
           Common formats: "07/2026", "AUG 2027", "08-26", "2026-12"
           It's often at the TOP or BOTTOM of the strip in smaller text.

        **SEARCH STRATEGY**:
        - Start by identifying the MAIN BRAND NAME (biggest text)
        - Look for dosage numbers after the brand (like 500, 650, 100, 400)
        - Scan the ENTIRE package carefully for dates
        - If text is rotated/sideways, mentally rotate it to read it

        OCR reference text (may be noisy):
        {ocr_text}

        Return ONLY this exact JSON format (no markdown, no explanation):
        {{"name": "BRAND NAME + DOSAGE", "expiry": "MM/YYYY or DATE"}}
        
        If you cannot find the field, return: {{"name": null, "expiry": null}}
        """

        models_to_try = [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",  # Keeping as fallback
        ]
        
        image_data = None
        if image_path and os.path.exists(image_path):
            try:
                # Resize image for better AI vision (don't send 10MB files)
                img = cv2.imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    max_dim = 1024
                    if h > max_dim or w > max_dim:
                        scale = max_dim / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    
                    # Encode to jpg buffer
                    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    image_data = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                print(f"[DEBUG] Image encoding failed: {e}")

        for model_name in models_to_try:
            # Try both v1beta and v1 endpoints
            for api_version in ["v1", "v1beta"]:
                url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={api_key}"
                headers = {"Content-Type": "application/json"}
                
                parts = []
                if image_data:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    })
                
                parts.append({"text": prompt})
                
                payload = {
                    "contents": [{
                        "parts": parts
                    }],
                    "generationConfig": {
                        "temperature": 0.1,
                        "topP": 0.95,
                        "topK": 40,
                        "maxOutputTokens": 1024,
                    }
                }
                
                try:
                    response = requests.post(url, headers=headers, json=payload, timeout=12)
                    if response.status_code == 200:
                        res_data = response.json()
                        if "candidates" in res_data and len(res_data["candidates"]) > 0:
                            cand = res_data["candidates"][0]
                            if "content" in cand and "parts" in cand["content"]:
                                res_text = cand["content"]["parts"][0]["text"].strip()
                                # Clean markdown if present
                                res_text = re.sub(r'```(?:json)?\s*|\s*```', '', res_text)
                                # Find JSON object even if there's surrounding text
                                json_match = re.search(r'\{[^{}]*\}', res_text, re.DOTALL)
                                if json_match:
                                    try:
                                        data = json.loads(json_match.group(0))
                                        
                                        name = data.get("name", None)
                                        expiry = data.get("expiry", None)
                                        
                                        # Clean up extracted values
                                        if name and isinstance(name, str) and name.strip().upper() not in ["NULL", "NONE", "UNKNOWN"]:
                                            name = name.strip().upper()
                                        else:
                                            name = None
                                            
                                        if expiry and isinstance(expiry, str) and expiry.strip().upper() not in ["NULL", "NONE", "UNKNOWN"]:
                                            expiry = expiry.strip().upper()
                                        else:
                                            expiry = None
                                        
                                        if name or expiry:
                                            print(f"[INFO] Gemini success using {model_name} ({api_version})")
                                            print(f"[INFO] Extracted: name={name}, expiry={expiry}")
                                            return {
                                                "medicine_name": name,
                                                "expiry_date": expiry
                                            }
                                    except json.JSONDecodeError as e:
                                        print(f"[DEBUG] JSON parse failed: {e}")
                                        continue
                    else:
                        print(f"[DEBUG] Gemini {model_name} ({api_version}) failed: {response.status_code}")
                        if response.status_code == 400:
                            print(f"[DEBUG] Body: {response.text}")
                except Exception as e:
                    print(f"[DEBUG] Gemini try ({model_name} {api_version}) failed: {e}")
                    continue
        
        return {"medicine_name": None, "expiry_date": None}


# Master Unified Extractor
class UnifiedAIExtractor:
    @staticmethod
    def extract_details(ocr_text: str, image_path: str = None) -> dict:
        """
        Tries Gemini (vision-based) or local improved extractor for details.
        Prioritizes image-based extraction when OCR quality is poor.
        """
        try:
            # 1. Try Gemini FIRST with minimal OCR context if image is available
            #    (Gemini is best at vision-based extraction from actual images)
            if image_path and os.path.exists(image_path):
                print("[INFO] Image available - prioritizing Gemini vision extraction...")
                # Send minimal OCR text so Gemini focuses on the image
                minimal_text = "Extract medicine name and expiry date from the image."
                res = GoogleGeminiExtractor.extract_fields(minimal_text, image_path=image_path)
                if res.get("medicine_name") or res.get("expiry_date"):
                    print(f"[INFO] Gemini vision extraction successful")
                    return res
            
            # 2. Try Gemini with full OCR context as fallback
            res = GoogleGeminiExtractor.extract_fields(ocr_text, image_path=image_path)
            if res.get("medicine_name") or res.get("expiry_date"):
                return res
            
            # 3. Try Local Improved Extractor as fallback (Reliable regex)
            print("[INFO] AI vision failed. Falling back to improved local regex...")
            from services.improved_field_extractor import ImprovedFieldExtractor
            
            local_res = {"medicine_name": None, "expiry_date": None}
            exp_date, _ = ImprovedFieldExtractor.extract_expiry_date(ocr_text)
            if exp_date:
                local_res["expiry_date"] = exp_date
                
            # For medicine name, we can try to find the most prominent line
            from services.improved_field_extractor import RegionSpecificExtractor
            med_name, _ = RegionSpecificExtractor.extract_from_top_region(ocr_text)
            if med_name:
                local_res["medicine_name"] = med_name
                
            return local_res
        except Exception as e:
            print(f"[ERROR] UnifiedAIExtractor failed: {e}")
            return {"medicine_name": None, "expiry_date": None}



    # Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Text Extraction Example")
    print("=" * 60)

    example_text = """
    ASPIRIN 500MG
    Strip of 10 Tablets

    Exp: 05/2026
    Use Before: May 2026

    Mfg: 01/2024

    Store in cool place
    """

    extractor = FieldExtractor()

    med_name, med_conf = extractor.extract_medicine_name(example_text)
    exp_date, exp_conf = extractor.extract_expiry_date(example_text)

    print(f"Medicine Name: {med_name} (confidence: {med_conf:.2%})")
    print(f"Expiry Date: {exp_date} (confidence: {exp_conf:.2%})")

    if exp_date:
        status = extractor.check_expiry_status(exp_date)
        print(f"Status: {status}")
