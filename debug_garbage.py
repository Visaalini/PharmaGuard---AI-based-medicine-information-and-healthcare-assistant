import re

def debug_is_garbage(text):
    """Debug version of _is_garbage"""
    if not text: 
        print("Empty text -> True")
        return True
    upper = text.strip().upper()
    print(f"\nDEBUG _is_garbage('{text}'):")
    
    # Check SKIP_LABELS
    SKIP_LABELS = [
        "TABLETS", "CAPSULES", "SYRUP", "INJECTION", "STRIP OF", "DMF", "API GRADE",
        "USP", "BP", "IP", "PHARMA", "LABORATORIES", "LIMITED", "LTD", "MFG", "EXP",
        "TAB", "CAP", "SACHET", "VIAL", "DROP", "LOTION", "CREAM", "OINTMENT", "GEL",
        "SEE", "SEES", "SIDE", "EFFECTS", "KEEP", "STORE", "REACH", "CHILDREN",
        "WARNING", "CAUTION", "SCHEDULE", "PRESCRIPTION", "PHARMACIST",
        "DOSAGE", "COMPOSITION", "EACH", "CONTAINS", "INGREDIENTS",
        "JUL", "JUN", "AUG", "SEP", "OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR", "MAY",
        "ETE", "ETT", "EII", "III", "LL", "OOO", "ZZZ", "XXX", "IEEE", "EEEE", "IIII", "OOOO", "UUUU",
        "WOR", "WOS", "OSE", "WOR OS EE", "WOROS"
    ]
    if upper in SKIP_LABELS: 
        print("  In SKIP_LABELS -> True")
        return True
    if upper.strip("0123456789 ").strip() in SKIP_LABELS:
        stripped = upper.strip("0123456789 ").strip()
        print(f"  Stripped '{upper}' -> '{stripped}' in SKIP_LABELS -> True")
        return True
    
    # Check date pattern
    if re.search(r'^(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|ETE|ETT|EII)\s*\d{1,2}$', upper):
        print("  Matches date pattern -> True")
        return True
    if len(upper) < 3:
        print("  Too short -> True")
        return True
    
    # Check space-separated pattern
    words = upper.split()
    if len(words) <= 3:
        short_vowel_words = sum(1 for w in words if len(w) <= 3 and not re.search(r'[AEIOUY]', w))
        print(f"  Short words check: {len(words)} words, {short_vowel_words} with no vowels")
        if short_vowel_words >= 2 and short_vowel_words >= len(words) * 0.5:
            print(f"    -> True (too many words without vowels)")
            return True
    
    # Check storage text
    if any(k in upper for k in [
        "STORE BELOW", "STORE AT", "KEEP IN", "COOL DRY", "PROTECT FROM",
        "REACH OF CHILDREN", "EACH", "CONTAINS", "COMPOSITION", "INGREDIENT",
        "DIRECTION", "DOSAGE", "WARNING", "CAUTION", "SCHEDULE",
        "PRESCRIPTION", "PHARMACIST", "MANUFACTURED", "MARKETED",
    ]):
        print("  Contains storage/instruction text -> True")
        return True

    if not re.search(r'[A-Z]', upper):
        print("  No uppercase letters -> True")
        return True

    # Vowel ratio
    vowels = re.findall(r'[AEIOUY]', upper)
    if len(upper) > 4:
        vowel_ratio = len(vowels) / len(upper)
        print(f"  Vowel ratio: {vowel_ratio:.3f} (length={len(upper)}, vowels={len(vowels)})")
        if vowel_ratio < 0.15:
            print(f"    -> True (too few vowels)")
            return True
    
    # Word level check
    for word in upper.split():
        if word in ["MFG", "EXP", "LOT", "BN", "ML", "MG", "MCG", "IU"]: 
            continue
        if re.match(r'^\d+.*$', word):
            print(f"  Skipping numeric word: {word}")
            continue
        if re.match(r'^.*\d+$', word):
            print(f"  Skipping alpha-numeric word: {word}")
            continue
        if len(word) >= 3 and not re.search(r'[AEIOUY]', word):
            print(f"  Word '{word}' has no vowels -> True")
            return True
    
    # Check repeating characters
    if re.search(r'(.)\1\1\1', upper):
        print("  Excessive repeating chars -> True")
        return True
    if re.search(r'([A-Z])\1\s([A-Z])\2\s([A-Z])\3', upper):
        print("  Pattern like 'EE EE EE' -> True")
        return True
    
    # Check pattern
    if re.match(r'^[A-Z]{2}\s+[A-Z]{2}(?:\s+[A-Z0-9]{2})?$', upper):
        print("  Matches 'EE EE 00' pattern -> True")
        return True

    # Too many symbols
    symbols = re.findall(r'[^A-Za-z0-9\s\-]', text)
    if len(symbols) > len(text) * 0.3:
        print(f"  Too many symbols ({len(symbols)}/{len(text)}) -> True")
        return True

    # Too short after cleaning
    clean = re.sub(r'[^A-Z]', '', upper)
    if len(clean) < 3:
        print(f"  Too short after cleaning ('{clean}') -> True")
        return True
    
    print("  -> False (PASSED)")
    return False

# Test
texts = ['DOLO 650 MG', 'WOR OS EE', 'CALPOL 500']
for text in texts:
    result = debug_is_garbage(text)
    print(f"Result: {result}\n")
