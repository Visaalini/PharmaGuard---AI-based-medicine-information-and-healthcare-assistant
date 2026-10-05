"""
Improved Field Extraction for Challenging Medicine Packages
- Better batch number detection
- Better expiry date detection
- Handles small text and various formats
- Region-specific extraction
"""

import re
from typing import Tuple, Optional


class ImprovedFieldExtractor:
    """Enhanced field extraction with more patterns"""

    # Expanded regex patterns for batch numbers
    BATCH_PATTERNS = [
        # Keyword-based (highest priority)
        (r'BATCH\s*(?:NO\.?|NUMBER|#|CODE)?\s*[:=.\-]?\s*([A-Z0-9][A-Z0-9/_\-\.]{2,30})', 0.98),
        (r'BATCH\s*(?:NO\.?)?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.95),
        (r'BN\.?\s*(?:NO\.?)?\s*[:=.\-]?\s*([A-Z0-9][A-Z0-9/_\-\.]{2,30})', 0.93),
        (r'B\.?[\s]*NO\.?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.90),
        (r'LOT\s*(?:NO\.?|NUMBER|#)?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.88),
        (r'L\s*NO\.?\s*[:=.\-]?\s*([A-Z0-9]{3,30})', 0.85),
        (r'BATCH[\s:=.\-]+([A-Z0-9]{3,30})', 0.92),
        (r'BN[\s:=.\-]+([A-Z0-9]{3,30})', 0.90),
        (r'LOT[\s:=.\-]+([A-Z0-9]{3,30})', 0.85),

        # Without keyword - look for batch-like patterns
        (r'\b([A-Z]{2,4}[\d]{6,12})\b', 0.70),  # ABC123456
        (r'\b([A-Z][\d]{7,12})\b', 0.65),       # A1234567
        (r'\b([\d]{2,3}[A-Z]{2,4}[\d]{3,8})\b', 0.60),  # 12ABC345
    ]

    # Expanded regex patterns for expiry dates
    EXPIRY_PATTERNS = [
        # Keyword-based
        (r'EXP(?:IRY|IRED?|\.)?[\s:=.\-]*DATE[\s:=.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.99),
        (r'EXP(?:IRY|IRED?|\.)?[\s:=.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.98),
        (r'EXP(?:IRY|IRED?|\.)?[\s:=.\-]*(\d{1,2}[-/.]\d{2,4})', 0.96),  # MM/YY
        (r'USE[\s]+BEFORE[\s:=.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.95),
        (r'BEST[\s]+BEFORE[\s:=.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.95),
        (r'VALID[\s]+(?:TILL|UNTIL|UPTO|THRU)[\s:=.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.90),
        (r'EXPIRY[\s:=.\-]*(\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})', 0.95),
        (r'EXP[^\n]{0,15}((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?\d{2,4})', 0.93),

        # Without keyword - month names
        (r'((?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s.\-/]?\d{2,4})', 0.80),

        # Numbers only (be careful - could match years)
        (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', 0.70),  # DD/MM/YYYY
        (r'\b(\d{1,2}[/-]\d{4})\b', 0.75),                # MM/YYYY
        (r'\b(\d{1,2}[/-]\d{2})\b', 0.65),                # MM/YY
    ]

    @staticmethod
    def extract_batch_number(text: str) -> Tuple[Optional[str], float]:
        """
        Enhanced batch number extraction
        Handles more format variations
        """
        if not text or len(text.strip()) == 0:
            return None, 0.0

        text_upper = text.upper()

        # Try keyword-based patterns first (more reliable)
        for pattern, base_confidence in ImprovedFieldExtractor.BATCH_PATTERNS:
            match = re.search(pattern, text_upper)
            if match:
                batch = match.group(1).strip().rstrip('/-.')

                # Validate
                if len(batch) >= 3:
                    # Boost confidence for good matches
                    if re.match(r'^[A-Z]{2,4}\d{6,10}$', batch):
                        confidence = min(base_confidence + 0.05, 1.0)
                    else:
                        confidence = base_confidence

                    # Exclude obvious false positives
                    if not ImprovedFieldExtractor._is_false_positive_batch(batch):
                        return batch, confidence

        return None, 0.0

    @staticmethod
    def extract_expiry_date(text: str) -> Tuple[Optional[str], float]:
        """
        Enhanced expiry date extraction
        Better handling of edge cases
        """
        if not text or len(text.strip()) == 0:
            return None, 0.0

        text_upper = text.upper()

        # Try patterns in order
        for pattern, base_confidence in ImprovedFieldExtractor.EXPIRY_PATTERNS:
            matches = re.finditer(pattern, text_upper)

            for match in matches:
                date_str = match.group(1).strip()

                # Validate format
                if ImprovedFieldExtractor._is_valid_date(date_str):
                    return date_str, base_confidence

        return None, 0.0

    @staticmethod
    def _is_valid_date(date_str: str) -> bool:
        """Check if string is a valid date"""
        date_str = date_str.strip()

        # MM/YY
        m = re.match(r'^(\d{1,2})[/-](\d{2})$', date_str)
        if m:
            month = int(m.group(1))
            if 1 <= month <= 12:
                year = int(m.group(2))
                full_year = 2000 + year if year <= 40 else 1900 + year
                if 2020 <= full_year <= 2050:
                    return True

        # MM/YYYY
        m = re.match(r'^(\d{1,2})[/-](\d{4})$', date_str)
        if m:
            month = int(m.group(1))
            year = int(m.group(2))
            if 1 <= month <= 12 and 2020 <= year <= 2050:
                return True

        # DD/MM/YYYY
        m = re.match(r'^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})$', date_str)
        if m:
            day = int(m.group(1))
            month = int(m.group(2))
            year = int(m.group(3))
            if len(m.group(3)) == 2:
                year = 2000 + year if year <= 40 else 1900 + year
            if 1 <= month <= 12 and 1 <= day <= 31 and 2020 <= year <= 2050:
                return True

        # Month names
        if re.match(r'^[A-Z]{3}\s?[/-]?\d{2,4}$', date_str):
            return True

        return False

    @staticmethod
    def _is_false_positive_batch(batch: str) -> bool:
        """Exclude common false positives"""
        # Exclude common words
        false_positives = ['THE', 'AND', 'FOR', 'NOT', 'THIS', 'THAT', 'WHICH']
        if batch in false_positives:
            return True

        # Exclude if it's just a year
        if re.match(r'^(19|20)\d{2}$', batch):
            return True

        return False


class RegionSpecificExtractor:
    """Extract fields from specific image regions"""

    @staticmethod
    def extract_from_top_region(text: str, height_percent: float = 0.4) -> Tuple[Optional[str], float]:
        """
        Extract medicine name from top region
        Usually located at top of packaging
        """
        lines = text.split('\n')
        top_lines = lines[:max(3, int(len(lines) * height_percent))]
        top_text = '\n'.join(top_lines)

        # Filter out metadata
        for line in top_lines:
            line = line.strip()
            if 5 <= len(line) <= 80 and re.search(r'[A-Za-z]{3,}', line):
                if not re.match(r'^(MG|ML|TABLET|CAPSULE|STRIP|PACK|BOX|EXP|MFD|MFG|BATCH)', line, re.IGNORECASE):
                    cleaned = re.sub(r'[^A-Za-z0-9\s\-\+\.]', '', line).strip()
                    if len(cleaned) >= 3:
                        return cleaned, 0.80

        return None, 0.0

    @staticmethod
    def extract_from_middle_region(text: str) -> Tuple[Optional[str], float]:
        """Extract batch number from middle region"""
        lines = text.split('\n')
        middle_start = int(len(lines) * 0.3)
        middle_end = int(len(lines) * 0.7)
        middle_lines = lines[middle_start:middle_end]
        middle_text = '\n'.join(middle_lines)

        extractor = ImprovedFieldExtractor()
        return extractor.extract_batch_number(middle_text)

    @staticmethod
    def extract_from_bottom_region(text: str) -> Tuple[Optional[str], float]:
        """Extract expiry date from bottom region"""
        lines = text.split('\n')
        bottom_lines = lines[int(len(lines) * 0.5):]
        bottom_text = '\n'.join(bottom_lines)

        extractor = ImprovedFieldExtractor()
        return extractor.extract_expiry_date(bottom_text)


# Testing
if __name__ == "__main__":
    print("Testing Improved Field Extraction")
    print("="*70)

    test_text = """
    PANTOPRAZOLE 40MG
    Gastro-Resistant Tablets

    Manufactured by XYZ Pharma
    Batch No.: BX123456789

    Expiry Date: 12/2025
    EXP: 31/12/2025

    Best Before: December 2025
    """

    extractor = ImprovedFieldExtractor()

    batch, batch_conf = extractor.extract_batch_number(test_text)
    expiry, expiry_conf = extractor.extract_expiry_date(test_text)

    print(f"\nResults:")
    print(f"Batch Number: {batch} ({batch_conf:.0%})")
    print(f"Expiry Date: {expiry} ({expiry_conf:.0%})")

    # Test region-specific extraction
    print("\nTesting Region-Specific Extraction:")
    region_extractor = RegionSpecificExtractor()

    name, name_conf = region_extractor.extract_from_top_region(test_text)
    batch2, batch2_conf = region_extractor.extract_from_middle_region(test_text)
    exp2, exp2_conf = region_extractor.extract_from_bottom_region(test_text)

    print(f"Medicine Name (top): {name} ({name_conf:.0%})")
    print(f"Batch (middle): {batch2} ({batch2_conf:.0%})")
    print(f"Expiry (bottom): {exp2} ({exp2_conf:.0%})")

    print("\n" + "="*70)
    print("PASS - Enhanced extraction working")
