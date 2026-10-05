
import sys
import os
import re

# Add the project root to sys.path
sys.path.append(r"d:\pp\PHARMAGUARD")

from models.predict import _extract_expiry, _normalize_date_format

test_cases = [
    "EXP: 08-26",
    "EXP:08/26",
    "E:08/26",
    "EXP. 08/2026",
    "EXPIRY 08/2026",
    "BEST BEFORE 08/2026",
    "USE BY 08/2026",
    "Valid upto 08/2026",
    "EXP 08 26",
    "08-26",
    "JUN 2026",
    "BATCH: AB123\nEXP: 08/26",
    "BATCH: AB123 EXP: 08/26",
    "EXP: 08-2 6", # Common OCR error
]

print("=== Testing Expiry Extraction ===")
for text in test_cases:
    result = _extract_expiry(text)
    print(f"Text: {text!r} -> Result: {result}")
