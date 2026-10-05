import os
import sys
from models.predict import predict_image
from dotenv import load_dotenv

load_dotenv()

def test_prediction(image_name):
    image_path = os.path.join("uploads", image_name)
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        return

    print(f"Testing prediction for: {image_path}")
    try:
        result = predict_image(image_path)
        print("\n--- Prediction Result ---")
        print(f"Verdict: {result.get('verdict')}")
        print(f"Medicine Name: {result.get('medicine_name')}")
        print(f"Expiry Date: {result.get('expiry_date')}")
        print(f"Expiry Status: {result.get('expiry_status')}")
        print(f"Confidence: {result.get('confidence')}%")
        print(f"OCR Confidence: {result.get('ocr_confidence')}%")
        print(f"Batch Number: {result.get('batch_number')}")
        print(f"Company: {result.get('company_name')}")
        # print(f"Extracted Text Snippet: {result.get('extracted_text')[:200]}...")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    test_image = "calpol-650mg-strip-of-15-tablets-2-1679306419.webp"
    test_prediction(test_image)
