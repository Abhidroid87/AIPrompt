"""
VERIFICATION & QUICK START GUIDE
For sickle cell CRISPR therapy QA system
"""

import os
from pathlib import Path

def verify_setup():
    """Verify all components are in place."""
    
    print("\n" + "="*70)
    print(" "*15 + "SYSTEM VERIFICATION CHECKLIST")
    print("="*70 + "\n")
    
    checks = []
    
    # Check 1: Sources folder with PDFs
    sources = Path("../Sources")
    if sources.exists():
        pdfs = list(sources.glob("*.pdf"))
        checks.append(("Sources folder", True, f"{len(pdfs)} PDFs found"))
        for pdf in pdfs:
            checks.append((f"  └─ {pdf.name}", True, ""))
    else:
        checks.append(("Sources folder", False, "Not found"))
    
    # Check 2: Data folder with text files
    data = Path("data")
    if data.exists():
        txts = list(data.glob("*.txt"))
        checks.append(("Data folder", True, f"{len(txts)} text files"))
        for txt in txts[:3]:
            checks.append((f"  └─ {txt.name}", True, ""))
    else:
        checks.append(("Data folder", False, "Not found"))
    
    # Check 3: Core modules
    modules = [
        "main_production.py",
        "rag_baseline.py",
        "gap_module.py",
        "kg_module.py",
        "mcts_module.py",
        "evaluator.py",
        "utils.py",
        "pdf_loader.py"
    ]
    
    for module in modules:
        path = Path(module)
        if path.exists():
            size = path.stat().st_size
            checks.append((f"Module: {module}", True, f"{size} bytes"))
        else:
            checks.append((f"Module: {module}", False, "Not found"))
    
    # Print results
    for check_name, status, details in checks:
        symbol = "✓" if status else "✗"
        status_text = "OK" if status else "MISSING"
        print(f"  [{symbol}] {check_name:<40} {status_text:<10} {details}")
    
    print("\n" + "="*70)
    print(" "*20 + "QUICK START COMMANDS")
    print("="*70 + "\n")
    
    print("1. Run production demo (complex sickle cell questions):")
    print("   python main_production.py\n")
    
    print("2. View results:")
    print("   cat results_production.csv\n")
    
    print("3. Extract PDF text:")
    print("   python -c \"import subprocess; subprocess.run(['pdftotext', '../Sources/Frangoul*.pdf', '-'])\"\n")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    verify_setup()
