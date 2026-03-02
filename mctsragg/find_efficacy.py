from pdf_parser import extract_text_from_pdf
import os
import re

sources_dir = '../Sources'
pdfs = [f for f in os.listdir(sources_dir) if f.endswith('.pdf')]

for pdf in pdfs:
    text = extract_text_from_pdf(os.path.join(sources_dir, pdf))
    print(f"\n=== FILE: {pdf} ===")
    
    # Look for 95% and 88%
    matches_95 = re.findall(r'([^.!?\n]*95%[^.!?\n]*)', text)
    matches_88 = re.findall(r'([^.!?\n]*88%[^.!?\n]*)', text)
    
    if matches_95:
        print("--- 95% Contexts ---")
        for m in matches_95[:5]:
            print(f"- {m.strip()}")
            
    if matches_88:
        print("--- 88% Contexts ---")
        for m in matches_88[:5]:
            print(f"- {m.strip()}")
