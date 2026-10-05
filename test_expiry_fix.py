import cv2
import numpy as np
import os
os.chdir(r"D:\PHARMAGUARD")

from detector import _extract_expiry_raw, _is_exp_keyword, detect_pharma

passed = 0
failed = 0

def check(label, result, expected):
    global passed, failed
    if result == expected:
        print(f"  PASS | {label} -> {result}")
        passed += 1
    else:
        print(f"  FAIL | {label} -> got {result!r}, expected {expected!r}")
        failed += 1

print("=== Problem 1: OCR misreads EXP keyword ===")
check("normal Exp", _extract_expiry_raw(["PARACIP-500", "Exp: 06/2026"]), "06/2026")
check("Fxp misread", _extract_expiry_raw(["PARACIP-500", "Fxp: 06/2026"]), "06/2026")
check("Bxp misread", _extract_expiry_raw(["PARACIP-500", "Bxp: 06/2026"]), "06/2026")
check("E×p misread", _extract_expiry_raw(["PARACIP-500", "E\u00d7p: 06/2026"]), "06/2026")
check("lowercase exp", _extract_expiry_raw(["PARACIP-500", "exp: 06/2026"]), "06/2026")
check("EXPIRY keyword", _extract_expiry_raw(["PARACIP-500", "EXPIRY: 06/2026"]), "06/2026")

print("\n=== Problem 2: Date on separate line ===")
check("keyword + next line", _extract_expiry_raw(["PARACIP-500", "Exp:", "06/2026"]), "06/2026")
check("Fxp + next line", _extract_expiry_raw(["PARACIP-500", "Fxp:", "06/2026"]), "06/2026")
check("USE BEFORE + next", _extract_expiry_raw(["PARACIP-500", "Use Before:", "06/2026"]), "06/2026")

print("\n=== Date format variants ===")
check("MM/YYYY", _extract_expiry_raw(["PARACIP-500", "Exp: 06/2026"]), "06/2026")
check("MM-YYYY", _extract_expiry_raw(["PARACIP-500", "Exp: 06-2026"]), "06-2026")
check("MM/YY", _extract_expiry_raw(["PARACIP-500", "Exp: 06/26"]), "06/26")
check("MMM YYYY", _extract_expiry_raw(["PARACIP-500", "Exp: JUN 2026"]), "JUN 2026")
check("DD/MM/YYYY", _extract_expiry_raw(["PARACIP-500", "Exp: 15/06/2026"]), "15/06/2026")

print("\n=== Fallback: no EXP keyword, date only ===")
check("date-only fallback", _extract_expiry_raw(["PARACIP-500", "Batch: AB123", "06/2026"]), "06/2026")

print(f"\n=== Summary: {passed} passed, {failed} failed ===")

print("\n=== Full detect_pharma() end-to-end ===")
img = np.ones((200, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, "PARACIP-500", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
cv2.putText(img, "EXP: 06/2026", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
cv2.imwrite("temp_test_expiry.png", img)

result = detect_pharma("temp_test_expiry.png")
print(f"  Status: {result['status']}")
print(f"  Medicine: {result['medicine_name']}")
print(f"  Expiry: {result['expiry_date']}")
e_ok = "PASS" if result["expiry_date"] is not None else "FAIL"
m_ok = "PASS" if result["medicine_name"] is not None else "FAIL"
print(f"  Expiry detected: {e_ok}")
print(f"  Medicine detected: {m_ok}")

os.remove("temp_test_expiry.png")
print(f"\nFinal: {passed} passed, {failed} failed")
