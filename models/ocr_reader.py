import cv2
import os
import shutil
import numpy as np

def _find_tesseract():
    env_path = os.environ.get("TESSERACT_CMD", "")
    if env_path and os.path.exists(env_path):
        return env_path

    win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.environ.get("USERNAME", "")),
        r"C:\Tesseract-OCR\tesseract.exe",
    ]
    for path in win_paths:
        if os.path.isfile(path):
            return path

    which_path = shutil.which("tesseract")
    if which_path:
        return which_path

    print("WARNING: Tesseract not found anywhere")
    return None


def _preprocess_for_ocr(img):
    """Apply fast preprocessing techniques and return variants."""
    results = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Upscale 2x (best balance of speed vs accuracy)
    h, w = gray.shape
    if w < 1000:
        upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = gray

    # 2. De-noise with bilateral filter (preserves edges while removing noise)
    denoised = cv2.bilateralFilter(upscaled, 9, 75, 75)
    results.append(denoised)

    # 3. Otsu threshold on denoised
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh)

    # 4. Adaptive threshold (handles shadows/uneven lighting)
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    results.append(adaptive)

    # 5. CLAHE contrast enhancement on denoised image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoised)
    results.append(clahe_img)

    # 6. Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    results.append(morph)

    # 7. Inverted (white text on dark background)
    inverted = cv2.bitwise_not(thresh)
    results.append(inverted)

    return results


def read_text(image_path):
    try:
        try:
            import pytesseract
        except ImportError:
            return "OCR unavailable - pytesseract not installed"

        tesseract_path = _find_tesseract()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            return "OCR unavailable - Tesseract not found"

        img = cv2.imread(image_path)
        if img is None:
            return "Image could not be loaded"

        # Use 3 fast configs (5 preprocessed × 3 configs = 15 attempts, fast)
        configs = [
            "--oem 3 --psm 6",   # uniform block of text
            "--oem 3 --psm 11",  # sparse text
            "--oem 3 --psm 3",   # fully automatic
        ]

        best_text = ""
        preprocessed_images = _preprocess_for_ocr(img)

        for processed in preprocessed_images:
            for cfg in configs:
                try:
                    text = pytesseract.image_to_string(processed, config=cfg)
                    text = text.strip()
                    if len(text) > len(best_text):
                        best_text = text
                        # Early exit: if we found good text, stop trying
                        if len(best_text) > 100:
                            break
                except Exception:
                    continue
            if len(best_text) > 100:
                break

        return best_text if best_text else "No text detected"

    except Exception as e:
        return f"OCR error - {str(e)}"