"""Debug script to diagnose OCR and extraction issues"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from models.predict import (
    _run_ocr, _extract_medicine_name_raw, _extract_expiry,
    _normalize, _text_layer, _fix_ocr_errors, _preprocess_image_variants
)
import cv2
import pytesseract
import re

def diagnose_image(image_path):
    """Diagnose OCR extraction for a given image"""
    
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {image_path}")
    print(f"{'='*70}\n")
    
    # Step 1: Raw OCR extraction
    print("[STEP 1] Running Raw OCR...")
    raw_text = _run_ocr(image_path)
    print(f"Raw OCR output ({len(raw_text)} chars):")
    print("-" * 70)
    print(raw_text)
    print("-" * 70)
    
    # Step 2: Normalized text
    print("\n[STEP 2] Text normalization...")
    norm_text = _normalize(raw_text)
    print(f"Normalized text ({len(norm_text)} chars):")
    print("-" * 70)
    print(norm_text)
    print("-" * 70)
    
    # Step 3: Medicine name extraction
    print("\n[STEP 3] Medicine name extraction...")
    med_name = _extract_medicine_name_raw(raw_text, image_path=image_path)
    print(f"Extracted medicine name: {med_name}")
    
    # Step 4: Expiry date extraction
    print("\n[STEP 4] Expiry date extraction...")
    expiry_date = _extract_expiry(raw_text, image_path=image_path)
    print(f"Extracted expiry date: {expiry_date}")
    
    # Step 5: Text layer analysis
    print("\n[STEP 5] Text layer (fuzzy matching) analysis...")
    tl = _text_layer(norm_text)
    print(f"  Matched name: {tl.get('matched_name')}")
    print(f"  Name matched: {tl.get('name_matched')}")
    print(f"  Company matched: {tl.get('company_matched')}")
    print(f"  Company name: {tl.get('company_name')}")
    
    # Step 6: Look for keywords
    print("\n[STEP 6] Keyword analysis...")
    text_upper = raw_text.upper()
    keywords = {
        'EXP': 'EXP' in text_upper,
        'BATCH': 'BATCH' in text_upper,
        'B.NO': 'B.NO' in text_upper,
        'MFG': 'MFG' in text_upper,
        'TABLET': 'TABLET' in text_upper,
        'CAPSULE': 'CAPSULE' in text_upper,
    }
    for kw, found in keywords.items():
        print(f"  {kw}: {found}")
    
    # Step 7: Raw image inspection
    print("\n[STEP 7] Preprocessing variants analysis...")
    variants = _preprocess_image_variants(image_path)
    print(f"  Number of preprocessing variants: {len(variants)}")
    for i, var in enumerate(variants):
        print(f"  Variant {i}: shape={var.shape}, dtype={var.dtype}")
    
    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    # Check for uploaded images
    upload_dir = "uploads"
    if os.path.exists(upload_dir):
        images = [f for f in os.listdir(upload_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
        if images:
            # Test the most recent image
            latest_image = os.path.join(upload_dir, images[-1])
            diagnose_image(latest_image)
        else:
            print("No images found in uploads folder")
    else:
        print("Uploads folder not found")
