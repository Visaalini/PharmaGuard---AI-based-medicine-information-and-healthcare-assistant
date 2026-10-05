"""Test the fixed prediction pipeline"""
from models.predict import predict_image, _is_ocr_failed, _is_garbage
import json

# Test on the problematic screenshot image
test_image = 'uploads/Screenshot_2025-09-17_194141.png'

print("\n" + "="*70)
print("TESTING FIXED PREDICTION PIPELINE")
print("="*70)

print(f"\nTesting image: {test_image}")

result = predict_image(test_image)

print("\n" + "-"*70)
print("RESULTS:")
print("-"*70)
print(json.dumps(result, indent=2))

print("\n" + "="*70)
print("KEY METRICS:")
print("="*70)
print(f"Status: {result.get('status')}")
print(f"Medicine Name: {result.get('medicine_name')}")
print(f"Expiry Date: {result.get('expiry_date')}")
print(f"Expiry Status: {result.get('expiry_status')}")
print(f"Verdict: {result.get('verdict')}")
print(f"OCR Confidence: {result.get('ocr_confidence'):.1f}%")
print(f"Overall Confidence: {result.get('confidence'):.1f}%")
