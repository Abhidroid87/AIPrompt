"""
Extract and analyze PDF sources for complex question generation.
"""

import pdfplumber
from pathlib import Path

def extract_pdf_content():
    """Extract text from all PDFs in Sources folder."""
    sources_dir = Path("../Sources")
    
    if not sources_dir.exists():
        print("Sources folder not found")
        return {}
    
    pdf_data = {}
    
    for pdf_file in sources_dir.glob("*.pdf"):
        print(f"\n{'='*70}")
        print(f"📄 Processing: {pdf_file.name}")
        print('='*70)
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                # Extract first 3 pages for analysis
                for i, page in enumerate(pdf.pages[:3]):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                pdf_data[pdf_file.name] = text
                
                # Print preview
                lines = text.split('\n')
                preview = '\n'.join(lines[:20])
                print(f"\n📝 Preview (first 20 lines):")
                print(preview)
                print(f"\n✓ Extracted from {len(pdf.pages)} pages total")
                
        except Exception as e:
            print(f"❌ Error reading {pdf_file.name}: {e}")
    
    return pdf_data

if __name__ == "__main__":
    data = extract_pdf_content()
    print(f"\n\n{'='*70}")
    print(f"Successfully extracted {len(data)} PDFs")
    print('='*70)
