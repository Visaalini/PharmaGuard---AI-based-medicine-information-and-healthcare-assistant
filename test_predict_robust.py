import os
import sys
from pathlib import Path

# Add the current directory to sys.path
sys.path.append(os.getcwd())

from models.predict import predict_image

def test_prediction(image_name):
    image_path = os.path.join("uploads", image_name)
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    print(f"\nTesting image: {image_name}")
    try:
        result = predict_image(image_path)
        print(f"  Medicine Name: {result['medicine_name']}")
        print(f"  Expiry Date  : {result['expiry_date']}")
        print(f"  Expiry Status: {result['expiry_status']}")
        print(f"  Verdict      : {result['verdict']}")
        print(f"  Confidence   : {result['confidence']}")
        print(f"  Status       : {result['status']}")
        print(f"  Extracted Text:\n{result['extracted_text']}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    # Test with a few images from uploads
    test_prediction("captured_image.jpg")
    test_prediction("000001.jpg")
    test_prediction("Screenshot_2025-09-17_173925.png")
