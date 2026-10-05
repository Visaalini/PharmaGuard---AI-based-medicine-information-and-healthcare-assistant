"""
YOLO + OCR for Medicine Detection
Combines YOLO object detection with OCR for better extraction
"""

import cv2
import numpy as np
import os


def detect_with_yolo(image_path, use_yolo=True):
    """
    Detect medicine using YOLO + OCR
    
    Args:
        image_path: Path to medicine image
        use_yolo: Whether to use YOLO (if available)
    
    Returns:
        dict: Detection results with YOLO bounding boxes and OCR text
    """
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("YOLO not installed. Run: pip install ultralytics")
        use_yolo = False
    
    result = {
        'yolo_detected': False,
        'yolo_boxes': [],
        'yolo_labels': [],
        'ocr_text': None,
        'expiry_date': None,
        'medicine_name': None,
        'method': 'OCR ONLY'
    }
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        result['error'] = 'Could not load image'
        return result
    
    if use_yolo:
        try:
            # Use YOLOv8 nano model (fastest)
            model = YOLO('yolov8n.pt')
            
            # Run detection
            yolo_result = model(image_path, save=False, verbose=False)
            
            # Get boxes and labels
            boxes = yolo_result[0].boxes
            if len(boxes) > 0:
                result['yolo_detected'] = True
                result['method'] = 'YOLO + OCR'
                
                # Extract box coordinates
                for box in boxes:
                    result['yolo_boxes'].append(box.xyxy[0].tolist())
                    result['yolo_labels'].append(model.names[int(box.cls[0])])
                
                print(f"[YOLO] Detected {len(boxes)} objects")
        
        except Exception as e:
            print(f"[YOLO] Error: {e}. Using OCR only.")
    
    # Run OCR on the full image
    ocr_text = run_ocr(image_path)
    result['ocr_text'] = ocr_text
    
    # Extract fields from OCR text
    try:
        from services.detection_service import (
            extract_medicine_name, extract_expiry_date
        )

        result['expiry_date'], result['expiry_status'] = extract_expiry_date(ocr_text)
        result['medicine_name'] = extract_medicine_name(ocr_text)
    except ImportError as e:
        result['expiry_date'] = None
        result['medicine_name'] = None
    
    return result


def run_ocr(image_path):
    """Run OCR on image with preprocessing"""
    try:
        import pytesseract
    except ImportError:
        return "OCR not available"
    
    # Set Tesseract path
    if os.name == 'nt':
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
    
    # Preprocess
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Upscale if small
    if gray.shape[1] < 1500:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # OCR with multiple configs
    configs = [
        '--oem 3 --psm 6',
        '--oem 3 --psm 11',
    ]
    
    best_text = ""
    for config in configs:
        try:
            text = pytesseract.image_to_string(gray, config=config)
            if len(text.strip()) > len(best_text):
                best_text = text
        except:
            continue
    
    return best_text.strip() if best_text else "No text detected"


def detect_with_yolo_crop(image_path):
    """
    Detect with YOLO and OCR on cropped regions
    More accurate but requires YOLO to detect regions first
    """
    try:
        from ultralytics import YOLO
    except:
        return detect_with_yolo(image_path, use_yolo=False)
    
    result = {
        'yolo_detected': False,
        'crop_results': [],
        'ocr_text': None,
        'expiry_date': None,
        'medicine_name': None,
    }
    
    img = cv2.imread(image_path)
    if img is None:
        return result
    
    try:
        # Use YOLO
        model = YOLO('yolov8n.pt')
        yolo_result = model(image_path, save=False, verbose=False)
        
        boxes = yolo_result[0].boxes
        if len(boxes) > 0:
            result['yolo_detected'] = True
            
            # Crop and OCR each detection
            h, w = img.shape[:2]
            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Ensure valid crop
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    crop = img[y1:y2, x1:x2]
                    crop_text = run_ocr_from_array(crop)
                    
                    result['crop_results'].append({
                        'box': [x1, y1, x2, y2],
                        'label': model.names[int(box.cls[0])],
                        'text': crop_text
                    })
    
    except Exception as e:
        print(f"YOLO crop error: {e}")
    
    # Also run full OCR
    full_text = run_ocr(image_path)
    result['ocr_text'] = full_text
    
    # Extract fields
    from services.detection_service import (
        extract_medicine_name, extract_expiry_date
    )

    result['expiry_date'], result['expiry_status'] = extract_expiry_date(full_text)
    result['medicine_name'] = extract_medicine_name(full_text)
    
    return result


def run_ocr_from_array(img_array):
    """Run OCR on numpy array"""
    try:
        import pytesseract
    except:
        return ""
    
    if img_array is None or img_array.size == 0:
        return ""
    
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) if len(img_array.shape) == 3 else img_array
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    try:
        text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
        return text.strip()
    except:
        return ""


# Test function
if __name__ == "__main__":
    import sys
    import os
    
    # Add project root to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    test_image = 'test_medicine_final.jpg'
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    
    print("=" * 60)
    print(f"YOLO + OCR Detection: {test_image}")
    print("=" * 60)
    
    result = detect_with_yolo(test_image)
    
    print(f"\nMethod: {result['method']}")
    print(f"YOLO Detected: {result['yolo_detected']}")
    if result['yolo_labels']:
        print(f"YOLO Labels: {result['yolo_labels']}")
    
    print(f"\n--- OCR Results ---")
    print(f"Medicine: {result.get('medicine_name', 'NOT DETECTED')}")
    print(f"Expiry:   {result.get('expiry_date', 'NOT DETECTED')}")
    print(f"Status:   {result.get('expiry_status', 'N/A')}")