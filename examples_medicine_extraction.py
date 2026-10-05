"""
COMPLETE WORKING EXAMPLE
- Shows how to use the extraction pipeline
- Step-by-step with input/output examples
- Shows integration with existing detection_service
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime


# ============================================================================
# EXAMPLE 1: Basic Field Extraction
# ============================================================================

def example_1_basic_extraction():
    """Example 1: Extract fields from OCR text"""
    from services.advanced_text_extraction import FieldExtractor

    print("\n" + "="*80)
    print("EXAMPLE 1: BASIC FIELD EXTRACTION FROM TEXT")
    print("="*80)

    # Simulated OCR output from medicine package
    ocr_text = """
    PARACETAMOL 500MG
    TABLET

    Manufactured by: XYZ Pharmaceuticals
    Batch No: AB123456789
    BN: AB123456789

    Expiry Date: 05/2026
    EXP: 05/2026

    Mfg Date: 01/2024

    Store in cool, dry place
    Keep away from children
    """

    extractor = FieldExtractor()

    # Extract fields
    medicine_name, name_conf = extractor.extract_medicine_name(ocr_text)
    batch_number, batch_conf = extractor.extract_batch_number(ocr_text)
    expiry_date, exp_conf = extractor.extract_expiry_date(ocr_text)
    expiry_status = extractor.check_expiry_status(expiry_date)

    # Display results
    print("\nINPUT (OCR Text):")
    print("-" * 80)
    print(ocr_text)

    print("\nOUTPUT (Extracted Fields):")
    print("-" * 80)
    print(f"Medicine Name: {medicine_name}")
    print(f"  └─ Confidence: {name_conf:.1%}")
    print(f"\nBatch Number: {batch_number}")
    print(f"  └─ Confidence: {batch_conf:.1%}")
    print(f"\nExpiry Date: {expiry_date}")
    print(f"  └─ Confidence: {exp_conf:.1%}")
    print(f"  └─ Status: {expiry_status}")

    result = {
        "medicine_name": medicine_name,
        "batch_number": batch_number,
        "expiry_date": expiry_date,
        "expiry_status": expiry_status,
        "confidences": {
            "name": f"{name_conf:.1%}",
            "batch": f"{batch_conf:.1%}",
            "expiry": f"{exp_conf:.1%}"
        }
    }

    print("\nJSON OUTPUT:")
    print(json.dumps(result, indent=2))

    return result


# ============================================================================
# EXAMPLE 2: Image Preprocessing Pipeline
# ============================================================================

def example_2_image_preprocessing():
    """Example 2: Preprocess medicine package image"""
    from services.image_preprocessor import ImagePreprocessor

    print("\n" + "="*80)
    print("EXAMPLE 2: IMAGE PREPROCESSING")
    print("="*80)

    # Create synthetic medicine package image
    print("\n[STEP 1] Creating synthetic medicine package image...")
    img = np.ones((600, 800, 3), dtype=np.uint8) * 240

    # Add text (simulating medicine packaging)
    cv2.putText(img, "ASPIRIN 500MG TABLET", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    cv2.putText(img, "Strip of 10 Tablets", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 1)
    cv2.putText(img, "Batch: AB987654321", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Exp: 12/2026", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, "Mfg: 12/2023", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Add some noise to simulate real conditions
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    test_image_path = "test_medicine_example.jpg"
    cv2.imwrite(test_image_path, img)
    print(f"✓ Test image created: {test_image_path}")

    # Preprocess
    print("\n[STEP 2] Preprocessing image...")
    preprocessor = ImagePreprocessor(debug=False)
    processed = preprocessor.preprocess_image(test_image_path)

    print(f"✓ Original image shape: {img.shape}")
    print(f"✓ Processed image shape: {processed.shape}")
    print(f"✓ Image prepared for OCR")

    # Save processed image
    cv2.imwrite("test_medicine_preprocessed.jpg", processed)
    print(f"✓ Preprocessed image saved: test_medicine_preprocessed.jpg")

    return {
        "original_image": test_image_path,
        "preprocessed_image": "test_medicine_preprocessed.jpg",
        "status": "success"
    }


# ============================================================================
# EXAMPLE 3: Complete Extraction Pipeline
# ============================================================================

def example_3_complete_pipeline():
    """Example 3: Run complete extraction pipeline"""
    from services.medicine_extraction_pipeline import MedicineDetailsExtractor

    print("\n" + "="*80)
    print("EXAMPLE 3: COMPLETE EXTRACTION PIPELINE")
    print("="*80)

    # Create test image
    print("\n[STEP 1] Creating test medicine package image...")
    img = np.ones((700, 1000, 3), dtype=np.uint8) * 245

    # Add text
    cv2.putText(img, "METFORMIN HCL 500MG", (60, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (20, 20, 20), 3)
    cv2.putText(img, "TABLET", (60, 160), cv2.FONT_HERSHEY_DUPLEX, 1.0, (40, 40, 40), 2)
    cv2.putText(img, "", (60, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 1)
    cv2.putText(img, "Batch No: BX789456123", (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Lot: BX789456123", (60, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(img, "", (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 1)
    cv2.putText(img, "Expiry: 08/2025", (60, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 150), 2)
    cv2.putText(img, "EXP Date: 31/08/2025", (60, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 100), 2)
    cv2.putText(img, "", (60, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 1)
    cv2.putText(img, "MFG Date: 08/2023", (60, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 1)

    test_image_path = "test_complete_medicine.jpg"
    cv2.imwrite(test_image_path, img)
    print(f"✓ Test image created: {test_image_path}")

    # Run extraction
    print("\n[STEP 2] Running extraction pipeline...")
    extractor = MedicineDetailsExtractor(use_roi_detection=True, debug=False)
    result = extractor.extract_medicine_details(test_image_path)

    # Display results
    print("\n[STEP 3] Results:")
    print("-" * 80)

    if result["success"]:
        extraction = result["extraction"]

        print(f"\n✓ EXTRACTION SUCCESSFUL")
        print(f"\nMedicine Name:")
        print(f"  Value: {extraction['medicine_name']['value']}")
        print(f"  Confidence: {extraction['medicine_name']['confidence']:.1%}")
        print(f"  Status: {extraction['medicine_name']['status']}")

        print(f"\nBatch Number:")
        print(f"  Value: {extraction['batch_number']['value']}")
        print(f"  Confidence: {extraction['batch_number']['confidence']:.1%}")
        print(f"  Status: {extraction['batch_number']['status']}")

        print(f"\nExpiry Date:")
        print(f"  Value: {extraction['expiry_date']['value']}")
        print(f"  Confidence: {extraction['expiry_date']['confidence']:.1%}")
        print(f"  Status: {extraction['expiry_date']['status']}")
        print(f"  Expiry Status: {extraction['expiry_date']['expiry_status']}")

        print(f"\nQuality Metrics:")
        metrics = result["quality_metrics"]
        print(f"  Overall Confidence: {metrics['overall_confidence']:.1%}")
        print(f"  Text Detected: {metrics['text_detected']}")
        print(f"  Regions Found: {metrics['regions_found']}")

    else:
        print(f"✗ EXTRACTION FAILED: {result['error']}")

    return result


# ============================================================================
# EXAMPLE 4: Integration with Existing Detection Service
# ============================================================================

def example_4_integration():
    """Example 4: How to integrate with existing detection_service.py"""

    print("\n" + "="*80)
    print("EXAMPLE 4: INTEGRATION WITH EXISTING DETECTION SERVICE")
    print("="*80)

    integration_code = '''
# FILE: services/detection_service.py (MODIFIED)

# Add this import at the top
from services.medicine_extraction_pipeline import MedicineDetailsExtractor

def detect_medicine(image_path, user_name=None, user_phone=None):
    """Original function - now enhanced with advanced extraction"""

    from models.predict import predict_image, get_confidence

    # Existing code for AI prediction
    prediction = predict_image(image_path)
    confidence = get_confidence(image_path)
    text = read_text(image_path)  # Existing OCR
    barcode = detect_barcode(image_path)

    # NEW: Advanced extraction with confidence scores
    try:
        extractor = MedicineDetailsExtractor(use_roi_detection=True, debug=False)
        extraction_result = extractor.extract_medicine_details(image_path)

        medicine_name = extraction_result["extraction"]["medicine_name"]["value"]
        batch_number = extraction_result["extraction"]["batch_number"]["value"]
        expiry_date = extraction_result["extraction"]["expiry_date"]["value"]
        expiry_status = extraction_result["extraction"]["expiry_date"]["expiry_status"]

        # Confidence scores for each field
        name_confidence = extraction_result["extraction"]["medicine_name"]["confidence"]
        batch_confidence = extraction_result["extraction"]["batch_number"]["confidence"]
        expiry_confidence = extraction_result["extraction"]["expiry_date"]["confidence"]

    except Exception as e:
        print(f"[WARNING] Advanced extraction failed: {e}")
        # Fallback to existing simple extraction
        medicine_name = extract_medicine_name(text)
        batch_number = extract_batch_number(text)
        expiry_date = extract_expiry_date(text)
        expiry_status = validate_expiry_status(expiry_date)
        name_confidence = 0.5
        batch_confidence = 0.5
        expiry_confidence = 0.5

    # Return enhanced result
    result = {
        "prediction": prediction,
        "confidence": confidence,
        "barcode": barcode,
        "medicine_name": medicine_name,
        "batch_number": batch_number,
        "expiry_date": expiry_date,
        "expiry_status": expiry_status,
        "extraction_confidence": {
            "name": name_confidence,
            "batch": batch_confidence,
            "expiry": expiry_confidence
        },
        "text": text
    }

    return result
    '''

    print("\nINTEGRATION STEPS:")
    print("-" * 80)
    print("""
1. Copy these files to your project:
   - services/image_preprocessor.py
   - services/advanced_text_extraction.py
   - services/medicine_extraction_pipeline.py

2. Modify services/detection_service.py:
   - Add import: from services.medicine_extraction_pipeline import MedicineDetailsExtractor
   - Update detect_medicine() function (see code below)

3. Install dependencies (if needed):
   - pip install easyocr
   - pip install opencv-python

4. Usage remains the same:
   - Call detect_medicine(image_path) as before
   - Get additional confidence scores in response

5. Response now includes:
   - medicine_name (string)
   - batch_number (string)
   - expiry_date (string)
   - expiry_status (VALID, EXPIRED, EXPIRING_SOON)
   - extraction_confidence (dict with name, batch, expiry scores)
    """)

    print("\nSAMPLE CODE:")
    print(integration_code)

    return {"status": "integration_example_shown"}


# ============================================================================
# EXAMPLE 5: Batch Processing Multiple Images
# ============================================================================

def example_5_batch_processing():
    """Example 5: Process multiple medicine package images"""
    from services.medicine_extraction_pipeline import MedicineDetailsExtractor

    print("\n" + "="*80)
    print("EXAMPLE 5: BATCH PROCESSING")
    print("="*80)

    # Create 3 test images
    test_images = []
    medicines = [
        ("ASPIRIN 500MG", "AB123456", "05/2026"),
        ("PARACETAMOL 650MG", "CD789012", "08/2025"),
        ("IBUPROFEN 400MG", "EF345678", "12/2026"),
    ]

    print("\n[STEP 1] Creating test images...")
    for i, (med_name, batch, expiry) in enumerate(medicines):
        img = np.ones((500, 700, 3), dtype=np.uint8) * 235

        cv2.putText(img, med_name, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(img, f"Batch: {batch}", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(img, f"EXP: {expiry}", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        path = f"test_medicine_{i+1}.jpg"
        cv2.imwrite(path, img)
        test_images.append(path)
        print(f"  ✓ Created: {path}")

    # Batch extract
    print("\n[STEP 2] Processing batch...")
    extractor = MedicineDetailsExtractor(use_roi_detection=False, debug=False)

    results = []
    for i, image_path in enumerate(test_images):
        print(f"\n  Processing {i+1}/{len(test_images)}: {image_path}")
        result = extractor.extract_medicine_details(image_path)
        results.append({
            "image": image_path,
            "medicine_name": result["extraction"]["medicine_name"]["value"],
            "batch_number": result["extraction"]["batch_number"]["value"],
            "expiry_date": result["extraction"]["expiry_date"]["value"],
        })

    # Summary
    print("\n[STEP 3] Batch Results Summary:")
    print("-" * 80)
    print(json.dumps(results, indent=2))

    return results


# ============================================================================
# MAIN: Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PHARMAGUARD - MEDICINE EXTRACTION PIPELINE - COMPLETE EXAMPLES")
    print("="*80)

    try:
        # Run each example
        print("\n\n" + "!"*80)
        print("Running Example 1...")
        print("!"*80)
        example_1_basic_extraction()

        print("\n\n" + "!"*80)
        print("Running Example 2...")
        print("!"*80)
        example_2_image_preprocessing()

        print("\n\n" + "!"*80)
        print("Running Example 3...")
        print("!"*80)
        example_3_complete_pipeline()

        print("\n\n" + "!"*80)
        print("Running Example 4...")
        print("!"*80)
        example_4_integration()

        print("\n\n" + "!"*80)
        print("Running Example 5...")
        print("!"*80)
        example_5_batch_processing()

        print("\n\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*80)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
