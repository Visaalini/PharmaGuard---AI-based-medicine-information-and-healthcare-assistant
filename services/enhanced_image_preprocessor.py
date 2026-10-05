"""
Enhanced Image Preprocessing for Low-Contrast Medicine Packaging
- Auto-detects and corrects rotation
- Handles glossy/reflective surfaces
- Improves small text visibility
- Multiple adaptive preprocessing strategies
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


class EnhancedImagePreprocessor:
    """Advanced preprocessing for challenging medicine package images"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.preprocessing_results = {}

    def preprocess_challenging_image(self, image_path: str) -> np.ndarray:
        """
        Enhanced pipeline for difficult medicine packaging images

        Handles:
        - Rotated/angled text
        - Low contrast
        - Small text
        - Glossy surfaces
        - Poor lighting

        Returns: Best preprocessed image for OCR
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: Auto-rotate if needed
        print("[STEP 1] Checking for rotation...")
        img = self._auto_correct_rotation(img)
        print("[OK] Rotation corrected")

        # Step 2: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 3: Handle glossy/reflective surfaces
        print("[STEP 2] Removing gloss/reflections...")
        deglossed = self._remove_gloss(gray)
        print("[OK] Gloss removed")

        # Step 4: Enhance small text (aggressive upscaling)
        print("[STEP 3] Upscaling for small text...")
        if gray.shape[1] < 1500:
            upscaled = cv2.resize(deglossed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        else:
            upscaled = deglossed
        print(f"[OK] Upscaled to {upscaled.shape}")

        # Step 5: Apply CLAHE for contrast
        print("[STEP 4] Enhancing contrast...")
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrasted = clahe.apply(upscaled)
        print("[OK] Contrast enhanced")

        # Step 6: Denoise
        print("[STEP 5] Denoising...")
        denoised = cv2.bilateralFilter(contrasted, 11, 95, 95)
        print("[OK] Denoised")

        # Step 7: Generate multiple thresholding variants
        print("[STEP 6] Generating threshold variants...")
        results = {
            'otsu': self._otsu_threshold(denoised),
            'adaptive': self._adaptive_threshold(denoised),
            'morphological': self._morphological_clean(denoised),
            'inverted': self._inverted_threshold(denoised),
            'sauvola': self._sauvola_threshold(denoised),
        }

        # Step 8: Deskew individual lines (for text detection)
        print("[STEP 7] Deskewing...")
        deskewed = self._deskew_image(results['adaptive'])
        results['deskewed'] = deskewed

        self.preprocessing_results = results

        # Return best variant (adaptive usually works best)
        best = results['adaptive']
        print(f"[OK] Using adaptive threshold")

        return best

    @staticmethod
    def _auto_correct_rotation(img: np.ndarray, max_rotation: float = 30) -> np.ndarray:
        """
        Auto-detect and correct rotation of image
        Useful for angled medicine packages
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Find lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)

        if lines is None or len(lines) == 0:
            return img

        # Get angles
        angles = []
        for line in lines[:30]:
            rho, theta = line[0]
            angle = theta * 180 / np.pi - 90
            if abs(angle) < max_rotation:
                angles.append(angle)

        if not angles:
            return img

        # Average angle
        avg_angle = np.median(angles)

        if abs(avg_angle) > 2:  # More than 2 degrees
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h),
                                borderMode=cv2.BORDER_REPLICATE)
            print(f"  Rotated by {avg_angle:.1f} degrees")

        return img

    @staticmethod
    def _remove_gloss(gray_img: np.ndarray) -> np.ndarray:
        """
        Remove glossy/reflective spots from medicine packaging
        Uses morphological operations
        """
        # Create bright spot mask
        _, bright_mask = cv2.threshold(gray_img, 240, 255, cv2.THRESH_BINARY)

        # Dilate to expand bright areas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        bright_mask = cv2.dilate(bright_mask, kernel, iterations=2)

        # Create gradient
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bright_gradient = cv2.dilate(bright_mask, kernel2, iterations=3)

        # Inpaint glossy areas
        result = cv2.inpaint(gray_img, bright_mask, 3, cv2.INPAINT_TELEA)

        return result

    @staticmethod
    def _otsu_threshold(gray_img: np.ndarray) -> np.ndarray:
        """Binary threshold using Otsu's method"""
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _adaptive_threshold(gray_img: np.ndarray) -> np.ndarray:
        """Adaptive threshold - best for varying lighting"""
        return cv2.adaptiveThreshold(
            gray_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 5  # Larger window for better results
        )

    @staticmethod
    def _morphological_clean(gray_img: np.ndarray) -> np.ndarray:
        """Clean using morphological operations"""
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Multiple iterations for better cleaning
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closed

    @staticmethod
    def _inverted_threshold(gray_img: np.ndarray) -> np.ndarray:
        """Inverted - for white text on dark background"""
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _sauvola_threshold(gray_img: np.ndarray) -> np.ndarray:
        """
        Sauvola thresholding - good for low contrast
        Implementation using OpenCV's adaptiveThreshold with different params
        """
        # Simulate Sauvola with adaptive threshold using different kernel
        return cv2.adaptiveThreshold(
            gray_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10
        )

    @staticmethod
    def _deskew_image(img: np.ndarray) -> np.ndarray:
        """
        Deskew image - rotates to make text horizontal
        Good for catching batch/expiry info at odd angles
        """
        coords = np.column_stack(np.where(img > 0))
        if len(coords) == 0:
            return img

        angle = cv2.minAreaRect(coords)[2]

        if angle < -45:
            angle = 90 + angle

        if abs(angle) > 0.5:
            h, w = img.shape
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            deskewed = cv2.warpAffine(img, matrix, (w, h))
            return deskewed

        return img

    def get_all_variants(self) -> Dict[str, np.ndarray]:
        """Get all preprocessing variants for multi-attempt OCR"""
        return self.preprocessing_results


class SmallTextEnhancer:
    """Specifically enhance small text on medicine packaging"""

    @staticmethod
    def extract_text_regions(preprocessed_img: np.ndarray) -> List[np.ndarray]:
        """
        Extract individual text regions for focused OCR
        Better for small, scattered text
        """
        contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter for text-like regions (small to medium)
            if 300 < area < 10000:  # Small text region
                aspect = w / h if h > 0 else 0
                if 0.5 < aspect < 15:  # Text-like aspect ratio
                    regions.append((x, y, w, h))

        # Sort by position (left to right, top to bottom)
        regions = sorted(regions, key=lambda r: (r[1], r[0]))

        return regions

    @staticmethod
    def enhance_text_region(img: np.ndarray, region: Tuple, padding: int = 20) -> np.ndarray:
        """
        Extract and enhance a specific text region

        Args:
            img: Original image
            region: (x, y, w, h) tuple
            padding: Padding around region

        Returns: Enhanced region image
        """
        x, y, w, h = region

        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)

        region_img = img[y_start:y_end, x_start:x_end]

        # Enhance region
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(region_img)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened


# Testing
if __name__ == "__main__":
    print("="*70)
    print("ENHANCED IMAGE PREPROCESSING FOR MEDICINE PACKAGING")
    print("="*70)

    # Create test image with small text and rotation
    test_img = np.ones((600, 800, 3), dtype=np.uint8) * 230

    # Add some "gloss" (bright spots)
    cv2.ellipse(test_img, (100, 100), (50, 30), 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(test_img, (700, 150), (40, 25), 0, 0, 360, (245, 245, 245), -1)

    # Add text at angle
    cv2.putText(test_img, "PANTOPRAZOLE 40MG", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (50, 50, 50), 2)
    cv2.putText(test_img, "Batch: B123456789", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    cv2.putText(test_img, "EXP: 12/2025", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)

    # Rotate slightly
    center = (400, 300)
    angle = 5
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    test_img = cv2.warpAffine(test_img, matrix, (800, 600))

    cv2.imwrite("test_challenging.jpg", test_img)
    print("\nCreated: test_challenging.jpg")

    # Process
    print("\nProcessing with enhanced preprocessor...")
    preprocessor = EnhancedImagePreprocessor(debug=True)
    processed = preprocessor.preprocess_challenging_image("test_challenging.jpg")

    print(f"\nProcessed shape: {processed.shape}")
    cv2.imwrite("test_challenging_processed.jpg", processed)
    print("Saved: test_challenging_processed.jpg")

    # Extract text regions
    print("\nExtracting text regions...")
    enhancer = SmallTextEnhancer()
    regions = enhancer.extract_text_regions(processed)
    print(f"Found {len(regions)} text regions")

    print("\n" + "="*70)
    print("SUCCESS")
    print("="*70)
