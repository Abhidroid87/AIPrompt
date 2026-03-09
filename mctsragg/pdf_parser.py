import pypdfium2 as pdfium
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pypdfium2."""
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        text = ""
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            textpage = page.get_textpage()
            text += textpage.get_text_range()
            text += "\n"
            textpage.close()
            page.close()
            
        pdf.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return ""

if __name__ == "__main__":
    content = extract_text_from_pdf("sample_sickle_cell.pdf")
    print(f"Extracted {len(content)} characters from PDF.")
    print("\nFirst 500 characters:")
    print(content[:500])
