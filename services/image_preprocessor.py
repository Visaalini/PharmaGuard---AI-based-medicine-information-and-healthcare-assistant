"""
Advanced Image Preprocessing for Medicine Package OCR
- Handles various lighting, angle, blur conditions
- Extracts text regions effectively
- Optimized for medicine packaging details
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import os

class ImagePreprocessor:
    """Advanced image preprocessing specifically for medicine packaging"""

    def __init__(self, debug=False):
        self.debug = debug
        self.preprocessing_results = {}

    def preprocess_image(self, image_path: str, target_width: int = 1280) -> np.ndarray:
        """
        Main preprocessing pipeline - optimized for medicine packaging

        Steps:
        1. Load and normalize image
        2. Auto-correct orientation
        3. Enhance contrast
        4. Denoise
        5. Threshold for text detection

        Returns: Best preprocessed image for OCR
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Step 1: Resize to standard size if needed
        if img.shape[1] > target_width:
            scale = target_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # Step 2: Auto-rotate if text is sideways (optional - skip if not needed)
        # img = self._auto_rotate(img)

        # Step 3: Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)

        # Step 5: Denoise - bilateral filter preserves edges
        denoised = cv2.bilateralFilter(clahe_img, 9, 75, 75)

        # Step 6: Upscale for better OCR (important!)
        if gray.shape[1] < 1000:
            upscaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        else:
            upscaled = denoised

        # Step 7: Generate multiple thresholding variants
        results = {
            'original': upscaled,
            'otsu': self._otsu_threshold(upscaled),
            'adaptive': self._adaptive_threshold(upscaled),
            'morphological': self._morphological_clean(upscaled),
            'inverted_otsu': self._inverted_threshold(upscaled),
        }

        self.preprocessing_results = results

        if self.debug:
            self._save_debug_images(image_path, results)

        # Return Otsu threshold by default (works well for medicine packaging)
        return results['otsu']

    def get_all_variants(self) -> Dict[str, np.ndarray]:
        """
        Get all preprocessing variants for multi-attempt OCR
        Useful when one variant fails - try others
        """
        return self.preprocessing_results

    @staticmethod
    def _otsu_threshold(gray_img: np.ndarray) -> np.ndarray:
        """Binary threshold using Otsu's method"""
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def _adaptive_threshold(gray_img: np.ndarray) -> np.ndarray:
        """Adaptive threshold - handles varying lighting"""
        return cv2.adaptiveThreshold(
            gray_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

    @staticmethod
    def _morphological_clean(gray_img: np.ndarray) -> np.ndarray:
        """Clean noise using morphological operations"""
        # First apply Otsu
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        return closed

    @staticmethod
    def _inverted_threshold(gray_img: np.ndarray) -> np.ndarray:
        """Inverted threshold - for white text on dark background"""
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(thresh)

    @staticmethod
    def _auto_rotate(img: np.ndarray) -> np.ndarray:
        """
        Auto-rotate image if text is rotated
        Uses angle detection from edges
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        edges = cv2.Canny(gray, 50, 150)

        # Find dominant angle
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        if lines is None or len(lines) == 0:
            return img

        angles = [line[0][1] for line in lines[:20]]
        avg_angle = np.median(angles) * 180 / np.pi

        # If angle is close to horizontal, it's probably text
        if abs(avg_angle) > 3:  # More than 3 degrees
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (w, h))

        return img

    def _save_debug_images(self, image_path: str, results: Dict):
        """Save debug images to disk"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        debug_dir = f"debug_preprocessing_{base_name}"

        os.makedirs(debug_dir, exist_ok=True)

        for name, img in results.items():
            cv2.imwrite(f"{debug_dir}/{name}.jpg", img)

        print(f"[DEBUG] Preprocessing images saved to: {debug_dir}")


class RegionOfInterestDetector:
    """
    Detect regions of interest (text regions) in medicine packaging
    Uses edge detection and contour finding
    """

    @staticmethod
    def find_text_regions(preprocessed_img: np.ndarray,
                         min_area: int = 500,
                         max_area: Optional[int] = None) -> List[Tuple]:
        """
        Find potential text regions in the image

        Returns: List of (x, y, w, h) for each detected region
        """
        # Find contours
        contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area
            if area < min_area:
                continue
            if max_area and area > max_area:
                continue

            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 10:  # Too square or too thin
                continue

            regions.append((x, y, w, h))

        # Sort by area (largest first)
        regions = sorted(regions, key=lambda r: r[2] * r[3], reverse=True)

        return regions

    @staticmethod
    def draw_regions(original_img: np.ndarray, regions: List[Tuple]) -> np.ndarray:
        """Draw detected regions on image for visualization"""
        img_copy = original_img.copy()

        for i, (x, y, w, h) in enumerate(regions[:10]):  # Show top 10
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_copy, f"Region {i+1}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img_copy

    @staticmethod
    def extract_region(original_img: np.ndarray, region: Tuple, padding: int = 10) -> np.ndarray:
        """
        Extract a specific region from the image with padding

        Args:
            original_img: Original color image
            region: (x, y, w, h) tuple
            padding: Padding around region (pixels)

        Returns: Extracted image region
        """
        x, y, w, h = region

        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(original_img.shape[1], x + w + padding)
        y_end = min(original_img.shape[0], y + h + padding)

        return original_img[y_start:y_end, x_start:x_end]


# Example usage
if __name__ == "__main__":
    # Example 1: Basic preprocessing
    print("=" * 60)
    print("Example 1: Image Preprocessing")
    print("=" * 60)

    preprocessor = ImagePreprocessor(debug=True)

    # Create a dummy image for testing
    dummy_img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    cv2.putText(dummy_img, "Aspirin 500mg", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    cv2.putText(dummy_img, "Batch: AB12345678", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(dummy_img, "EXP: 05/2026", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    cv2.imwrite("test_medicine.jpg", dummy_img)

    # Preprocess
    processed = preprocessor.preprocess_image("test_medicine.jpg")
    print(f"✓ Processed image shape: {processed.shape}")

    # Example 2: Region detection
    print("\n" + "=" * 60)
    print("Example 2: Text Region Detection")
    print("=" * 60)

    roi_detector = RegionOfInterestDetector()
    regions = roi_detector.find_text_regions(processed)
    print(f"✓ Found {len(regions)} text regions")

    for i, region in enumerate(regions[:5]):
        print(f"  Region {i+1}: {region}")

    # Visualize regions
    visualized = roi_detector.draw_regions(dummy_img, regions)
    cv2.imwrite("test_regions_detected.jpg", visualized)
    print(f"✓ Visualization saved: test_regions_detected.jpg")
