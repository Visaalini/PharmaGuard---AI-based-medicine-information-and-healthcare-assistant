"""Quick test: Run OCR + expiry extraction on a real image."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.predict import _run_ocr, _extract_expiry, predict_image

# Pick the first real image
real_dir = os.path.join("dataset", "real")
images = [f for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
if not images:
    print("No images found in dataset/real/")
    sys.exit(1)

test_img = os.path.join(real_dir, images[0])
print(f"Testing: {test_img}")
print()

# Test 1: OCR only
print("=" * 60)
print("OCR EXTRACTED TEXT:")
print("=" * 60)
text = _run_ocr(test_img)
print(text if text else "(empty)")
print()

# Test 2: Expiry extraction
print("=" * 60)
print("EXPIRY DATE EXTRACTION:")
print("=" * 60)
expiry = _extract_expiry(text, image_path=test_img)
print(f"Expiry: {expiry if expiry else 'Not detected'}")
print()

# Test 3: Full pipeline
print("=" * 60)
print("FULL PREDICTION PIPELINE:")
print("=" * 60)
result = predict_image(test_img)
print(f"Verdict:      {result.get('verdict')}")
print(f"Medicine:     {result.get('medicine_name')}")
print(f"Expiry:       {result.get('expiry_date')}")
print(f"Exp Status:   {result.get('expiry_status')}")
print(f"Batch:        {result.get('batch_number')}")
print(f"CNN Status:   {result.get('status')}")
print(f"OCR Conf:     {result.get('ocr_confidence')}%")
print(f"Confidence:   {result.get('confidence')}%")
print(f"Conf Label:   {result.get('confidence_label')}")
