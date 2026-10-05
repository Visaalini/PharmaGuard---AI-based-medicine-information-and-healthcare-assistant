#!/usr/bin/env python3
"""Test improved Gemini extraction on clear medicine images"""

import sys
import os
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.predict import predict_image
import json

# Test images in order of preference
TEST_IMAGES = [
    "uploads/medicine_01.jpg",       # Real medicine package image
    "uploads/medicine_06.jpg",       # Real medicine package image  
    "uploads/images01.jpg",          # JPG medicine image
    "uploads/000001.jpg",            # Numbered dataset image
    "uploads/Screenshot_2025-09-17_194141.png",  # Original problematic screenshot
]

print("=" * 80)
print("TESTING IMPROVED GEMINI EXTRACTION ON CLEAR MEDICINE IMAGES")
print("=" * 80)

for test_image in TEST_IMAGES:
    if not os.path.exists(test_image):
        print(f"\n[SKIP] Image not found: {test_image}")
        continue
    
    print(f"\n{'=' * 80}")
    print(f"Testing: {test_image}")
    print(f"{'=' * 80}")
    
    try:
        result = predict_image(test_image)
        
        # Pretty print results
        print("\nRESULTS:")
        print(json.dumps({
            "status": result.get("status"),
            "confidence": result.get("confidence"),
            "verdict": result.get("verdict"),
            "medicine_name": result.get("medicine_name"),
            "expiry_date": result.get("expiry_date"),
            "batch_number": result.get("batch_number"),
            "ocr_confidence": result.get("ocr_confidence"),
        }, indent=2))
        
    except Exception as e:
        print(f"[ERROR] Failed to process {test_image}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
