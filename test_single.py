import os
import sys
from pathlib import Path

sys.path.append(os.getcwd())
from models.predict import predict_image

res = predict_image('uploads/captured_image.jpg')
print(f"Name: {res['medicine_name']}")
print(f"Expiry: {res['expiry_date']}")
print(f"Verdict: {res['verdict']}")
print(f"Confidence: {res['confidence']}")
