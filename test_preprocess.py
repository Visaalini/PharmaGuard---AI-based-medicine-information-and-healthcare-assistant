import cv2
import pytesseract
import numpy as np

img = cv2.imread('uploads/Screenshot_2025-09-17_194141.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Upscale
h, w = gray.shape[:2]
scale = 3000.0 / w if w < 1500 else 1
gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast = clahe.apply(gray)

# Otsu threshold
_, otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Try different preprocessing variants
variants = [gray, contrast, otsu]
configs = ["--psm 6 --oem 3", "--psm 11 --oem 3"]

for i, var in enumerate(variants):
    for cfg in configs:
        text = pytesseract.image_to_string(var, config=cfg)
        if text.strip():
            print(f"Variant {i}, Config '{cfg}':")
            print(repr(text[:300]))
            print()
