import re
import sys
import os

def verify_latex(file_path):
    print(f"Checking LaTeX file: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return False

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    errors = []

    # 1. Broken citations
    if '? ]' in content or '[?' in content:
        errors.append("Found unresolved citation patterns like '? ]' or '[?'")

    # 2. Non-compliant terms (garantit, certifie)
    # Ignore comments when searching
    lines = content.split('\n')
    for idx, line in enumerate(lines, 1):
        clean_line = line.split('%')[0] # Remove comments
        
        # Forbidden words
        if re.search(r'\bgarantit[a-z]*\b', clean_line, re.IGNORECASE):
            errors.append(f"Line {idx}: Found prohibited word related to 'garantit': '{line.strip()}'")
        if re.search(r'\bcertifie[a-z]*\b', clean_line, re.IGNORECASE):
            errors.append(f"Line {idx}: Found prohibited word related to 'certifie': '{line.strip()}'")
        if re.search(r'endommagement\s+progressif', clean_line, re.IGNORECASE):
            errors.append(f"Line {idx}: Found prohibited phrase 'endommagement progressif': '{line.strip()}'")
            
        # Local Windows paths
        if re.search(r'\b[C-DF-Z]:[/\\]', clean_line, re.IGNORECASE):
            errors.append(f"Line {idx}: Found absolute local path pattern: '{line.strip()}'")

    # 3. Cross-explanation check for 128 and 384
    has_128 = "128" in content
    has_384 = "384" in content
    if not (has_128 and has_384):
        errors.append(f"Cross-explanation check failed: '128' present = {has_128}, '384' present = {has_384}. Both must be present to explain dataset differences.")

    if errors:
        print("\n--- LaTeX Verification Errors Found ---")
        for error in errors:
            print(f"[ERROR] {error}")
        return False
    else:
        print("\n[SUCCESS] LaTeX file passed all verification checks.")
        return True

if __name__ == "__main__":
    target = os.path.join("livrables", "main_public_fr.tex")
    if len(sys.argv) > 1:
        target = sys.argv[1]
    
    success = verify_latex(target)
    sys.exit(0 if success else 1)
