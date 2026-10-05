"""
Complete Medicine Package Information Extraction Pipeline
Integrates:
- Image preprocessing
- Text extraction (EasyOCR + Tesseract)
- Field extraction (Name, Batch, Expiry)
- ROI detection
- Confidence scoring
"""

import cv2
import numpy as np
from typing import Dict, Optional, List, Tuple
import json
from datetime import datetime

from services.image_preprocessor import ImagePreprocessor, RegionOfInterestDetector
from services.advanced_text_extraction import AdvancedTextExtractor, FieldExtractor


class MedicineDetailsExtractor:
    """Main extraction pipeline for medicine package details"""

    def __init__(self, use_roi_detection: bool = True, debug: bool = False):
        self.preprocessor = ImagePreprocessor(debug=debug)
        self.roi_detector = RegionOfInterestDetector()
        self.text_extractor = AdvancedTextExtractor(use_easyocr=True, use_tesseract=True)
        self.field_extractor = FieldExtractor()
        self.use_roi_detection = use_roi_detection
        self.debug = debug
        self.extraction_stats = {}

    def extract_medicine_details(self, image_path: str) -> Dict:
        """
        Complete extraction pipeline

        Process:
        1. Load and preprocess image
        2. (Optional) Detect text regions
        3. Extract all text using OCR
        4. Parse text to extract fields
        5. Validate and score results
        6. Return structured data

        Returns: Dictionary with all extracted information
        """
        print(f"\n{'='*70}")
        print(f"EXTRACTING MEDICINE DETAILS FROM: {image_path}")
        print(f"{'='*70}")

        try:
            # Step 1: Load and preprocess image
            print("\n[STEP 1] Preprocessing image...")
            original_img = cv2.imread(image_path)
            if original_img is None:
                return self._error_response("Could not load image")

            preprocessed_img = self.preprocessor.preprocess_image(image_path)
            print(f"[OK] Image preprocessed: {preprocessed_img.shape}")

            # Step 2: Detect text regions (optional)
            regions = []
            if self.use_roi_detection:
                print("\n[STEP 2] Detecting text regions...")
                regions = self.roi_detector.find_text_regions(preprocessed_img, min_area=500)
                print(f"✓ Found {len(regions)} text regions")

            # Step 3: Extract text from image
            print("\n[STEP 3] Extracting text with OCR...")
            extracted_text = self.text_extractor.extract_text_from_image(image_path, preprocessed_img)
            print(f"✓ Extracted {len(extracted_text.split())} words from image")
            print(f"\nExtracted text preview:\n{extracted_text[:300]}...")

            # Step 4: Parse fields
            print("\n[STEP 4] Parsing medicine details...")
            medicine_name, name_conf = self.field_extractor.extract_medicine_name(extracted_text)
            batch_number, batch_conf = self.field_extractor.extract_batch_number(extracted_text)
            expiry_date, exp_conf = self.field_extractor.extract_expiry_date(extracted_text)

            print(f"✓ Medicine Name: {medicine_name} ({name_conf:.1%} confidence)")
            print(f"✓ Batch Number: {batch_number} ({batch_conf:.1%} confidence)")
            print(f"✓ Expiry Date: {expiry_date} ({exp_conf:.1%} confidence)")

            # Step 5: Check expiry status
            print("\n[STEP 5] Validating expiry status...")
            expiry_status = self.field_extractor.check_expiry_status(expiry_date)
            print(f"✓ Expiry Status: {expiry_status}")

            # Step 6: Compile results
            print("\n[STEP 6] Compiling results...")
            result = {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "extraction": {
                    "medicine_name": {
                        "value": medicine_name,
                        "confidence": round(name_conf, 3),
                        "status": "high_confidence" if name_conf > 0.75 else "medium_confidence" if name_conf > 0.5 else "low_confidence"
                    },
                    "batch_number": {
                        "value": batch_number,
                        "confidence": round(batch_conf, 3),
                        "status": "high_confidence" if batch_conf > 0.75 else "medium_confidence" if batch_conf > 0.5 else "low_confidence"
                    },
                    "expiry_date": {
                        "value": expiry_date,
                        "confidence": round(exp_conf, 3),
                        "status": "high_confidence" if exp_conf > 0.75 else "medium_confidence" if exp_conf > 0.5 else "low_confidence",
                        "expiry_status": expiry_status
                    }
                },
                "quality_metrics": {
                    "overall_confidence": round((name_conf + batch_conf + exp_conf) / 3, 3),
                    "text_detected": len(extracted_text.split()) > 0,
                    "regions_found": len(regions),
                    "preprocessed_successfully": preprocessed_img is not None
                },
                "raw_text": extracted_text[:1000],  # First 1000 chars for reference
                "regions": regions[:10] if regions else []  # Top 10 regions
            }

            print(f"\n{'='*70}")
            print("SUCCESS: EXTRACTION COMPLETE")
            print(f"{'='*70}")

            return result

        except Exception as e:
            print(f"\n[ERROR] ERROR: {str(e)}")
            return self._error_response(str(e))

    def extract_with_roi_focus(self, image_path: str, region_index: int = 0) -> Dict:
        """
        Extract from specific text region (ROI)
        Useful when full-image extraction has too much noise
        """
        print(f"\n[ROI EXTRACTION] Focusing on region {region_index}...")

        try:
            original_img = cv2.imread(image_path)
            preprocessed_img = self.preprocessor.preprocess_image(image_path)

            # Find regions
            regions = self.roi_detector.find_text_regions(preprocessed_img)
            if region_index >= len(regions):
                return self._error_response(f"Region {region_index} not found (total: {len(regions)})")

            # Extract specific region
            region = regions[region_index]
            region_img = self.roi_detector.extract_region(original_img, region, padding=10)

            # Save temporary region for OCR
            temp_path = "temp_region.jpg"
            cv2.imwrite(temp_path, region_img)

            # Extract text from region
            text = self.text_extractor.extract_text_from_image(temp_path)

            # Parse fields
            medicine_name, _ = self.field_extractor.extract_medicine_name(text)
            batch_number, _ = self.field_extractor.extract_batch_number(text)
            expiry_date, _ = self.field_extractor.extract_expiry_date(text)

            return {
                "success": True,
                "region": region_index,
                "medicine_name": medicine_name,
                "batch_number": batch_number,
                "expiry_date": expiry_date,
                "text": text
            }

        except Exception as e:
            return self._error_response(str(e))

    @staticmethod
    def _error_response(error_msg: str) -> Dict:
        """Generate error response"""
        return {
            "success": False,
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
            "extraction": {
                "medicine_name": {"value": None, "confidence": 0.0},
                "batch_number": {"value": None, "confidence": 0.0},
                "expiry_date": {"value": None, "confidence": 0.0}
            }
        }

    def batch_extract(self, image_paths: List[str]) -> List[Dict]:
        """Process multiple images"""
        results = []
        for image_path in image_paths:
            result = self.extract_medicine_details(image_path)
            results.append(result)
        return results

    def get_extraction_stats(self) -> Dict:
        """Get extraction statistics"""
        return self.extraction_stats


# ============================================================================
# Integration with existing detection_service.py
# ============================================================================

def enhanced_detect_medicine(image_path: str, user_name: str = None) -> Dict:
    """
    Enhanced version of detect_medicine() with advanced extraction

    Drop-in replacement for services.detection_service.detect_medicine()
    """
    from models.predict import predict_image, get_confidence
    from services.detection_service import detect_barcode

    try:
        # Get AI prediction (existing)
        prediction = predict_image(image_path)
        confidence = get_confidence(image_path)
        barcode = detect_barcode(image_path)

        # NEW: Advanced extraction pipeline
        extractor = MedicineDetailsExtractor(use_roi_detection=True, debug=False)
        extraction_result = extractor.extract_medicine_details(image_path)

        # Combine results
        result = {
            "prediction": prediction,
            "confidence": confidence,
            "barcode": barcode,
            "medicine_name": extraction_result["extraction"]["medicine_name"]["value"],
            "batch_number": extraction_result["extraction"]["batch_number"]["value"],
            "expiry_date": extraction_result["extraction"]["expiry_date"]["value"],
            "expiry_status": extraction_result["extraction"]["expiry_date"].get("expiry_status"),
            "extracted_text": extraction_result.get("raw_text"),
            "quality_metrics": extraction_result.get("quality_metrics", {}),
            "extraction_confidence": {
                "name": extraction_result["extraction"]["medicine_name"]["confidence"],
                "batch": extraction_result["extraction"]["batch_number"]["confidence"],
                "expiry": extraction_result["extraction"]["expiry_date"]["confidence"]
            }
        }

        return result

    except Exception as e:
        print(f"[ERROR] Enhanced detection failed: {e}")
        # Fallback to original detection service
        from services.detection_service import detect_medicine
        return detect_medicine(image_path, user_name)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MEDICINE PACKAGE EXTRACTION PIPELINE - DEMO")
    print("="*70)

    # Example 1: Single image extraction
    print("\n[EXAMPLE 1] Single Image Extraction")
    print("-" * 70)

    # Create a test image
    test_img = np.ones((600, 800, 3), dtype=np.uint8) * 220
    cv2.putText(test_img, "ASPIRIN 500mg", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(test_img, "Batch No: AB12345678", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_img, "EXP: 05/2026", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_img, "Mfg: 01/2024", (50, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_img, "Strip of 10 Tablets", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    test_path = "test_medicine_package.jpg"
    cv2.imwrite(test_path, test_img)

    # Extract
    extractor = MedicineDetailsExtractor(use_roi_detection=True, debug=True)
    result = extractor.extract_medicine_details(test_path)

    # Display results
    print("\nRESULTS:")
    print(json.dumps(result, indent=2))

    # Example 2: Batch extraction
    print("\n[EXAMPLE 2] Batch Processing")
    print("-" * 70)
    print("(Skipped for demo - would process multiple images)")

    print("\n" + "="*70)
    print("✓ DEMO COMPLETE")
    print("="*70)
