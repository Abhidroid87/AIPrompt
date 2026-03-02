"""
PDF-powered MCTS-RAG-KG system for Sickle Cell Disease research.
Loads from Sources folder and generates complex multi-step questions.
"""

import os
from pathlib import Path

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def extract_pdf_texts():
    """Extract text from PDFs in Sources folder."""
    sources_dir = Path("../Sources")
    all_texts = []
    
    if not sources_dir.exists():
        print("⚠️  Sources folder not found")
        return []
    
    # Try with pdftotext system command (faster, no dependencies)
    import subprocess
    for pdf_file in sorted(sources_dir.glob("*.pdf")):
        try:
            result = subprocess.run(
                ['pdftotext', str(pdf_file), '-'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.stdout:
                all_texts.append(f"[Source: {pdf_file.name}]\n{result.stdout}")
        except Exception as e:
            print(f"⚠️  Could not extract {pdf_file.name}: {e}")
    
    return all_texts

# ==================== COMPLEX MULTI-STEP QUESTIONS ====================
# Based on actual PDF content: CTX001, CASGEVY (exa-cel), clinical trials
COMPLEX_QUESTIONS = [
    # Q1: Mechanism + Safety + Efficacy
    {
        "question": "How does CRISPR-Cas9 gene editing with BCL11A targeting work to treat sickle cell disease, and what was the efficacy and safety profile in the CLIMB-121 clinical trial?",
        "complexity": "Requires understanding: BCL11A mechanism, CRISPR-Cas9 technology, fetal hemoglobin reactivation, clinical outcomes (97% VOC-free), and safety data",
        "domain_concepts": ["BCL11A", "CRISPR-Cas9", "ex vivo gene editing", "fetal hemoglobin", "CD34+ HSPCs", "vaso-occlusive crises", "myeloablative conditioning", "clinical efficacy"]
    },
    
    # Q2: Comparing therapies
    {
        "question": "What are the key differences between CTX001 and exagamglogene autotemcel (CASGEVY) approaches for sickle cell disease treatment, and how do their clinical outcomes compare?",
        "complexity": "Requires: knowledge of two different therapies, their mechanisms, clinical trial data, outcomes (VOC-free rates, transfusion independence), and regulatory status",
        "domain_concepts": ["CTX001", "CASGEVY", "exagamglogene", "gene editing", "clinical trials", "VOC-free", "transfusion independence"]
    },
    
    # Q3: Trade-offs and risks
    {
        "question": "What are the benefits and risks of CRISPR-Cas9 based gene therapy for sickle cell disease? Discuss myeloablative conditioning, off-target effects, and long-term safety considerations.",
        "complexity": "Requires: multi-faceted reasoning about benefits (permanent cure), risks (conditioning toxicity, unknown long-term effects), regulatory approval basis",
        "domain_concepts": ["myeloablative busulfan", "off-target effects", "HLH", "serious adverse events", "durable response", "functional cure", "risk-benefit"]
    },
    
    # Q4: Population specifics
    {
        "question": "Who are the eligible candidates for CRISPR gene therapy in the CLIMB trials, and why are patients with a history of recurrent vaso-occlusive crises specifically targeted?",
        "complexity": "Requires: eligibility criteria analysis, understanding disease severity, treatment selection rationale, patient stratification",
        "domain_concepts": ["age 12-35", "recurrent VOCs", "severe SCD", "patient selection", "trial enrollment", "inclusion criteria"]
    },
    
    # Q5: Mechanism explanation with evidence
    {
        "question": "Explain how ex vivo CRISPR-Cas9 editing of CD34+ hematopoietic stem cells can achieve 97% freedom from vaso-occlusive crises in sickle cell patients. What evidence supports this efficacy?",
        "complexity": "Requires: mechanistic understanding of how fetal hemoglobin levels (39.6% to 49.6%) translate to clinical benefit, evidence-based reasoning",
        "domain_concepts": ["CD34+", "HSPCs", "hemoglobin levels", "fetal hemoglobin", "97% efficacy", "clinical evidence", "mechanism-outcome link"]
    },
]

# ==================== VALIDATION METRICS ====================
VALIDATION_SETUP = {
    "test_questions": COMPLEX_QUESTIONS,
    "primary_metric": "gap_reduction",  # Main feature: information gap reduction
    "secondary_metrics": [
        "answer_comprehensiveness",  # Does it cover all aspects?
        "faithfulness",              # Is it supported by sources?
        "kg_entity_extraction",      # How many entities extracted?
        "mcts_improvement",          # Does MCTS help over baseline?
    ],
    "computation_constraints": {
        "max_mcts_iterations": 2,
        "max_rag_retrievals": 3,
        "timeout_per_query": 30,
        "embedding_batch_size": 5,
    }
}

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  SICKLE CELL CRISPR THERAPY RESEARCH QUESTIONS")
    print("="*70)
    
    # Check PDF sources
    pdf_texts = extract_pdf_texts()
    print(f"\n✓ Loaded {len(pdf_texts)} PDF sources from Sources folder")
    
    # Show questions
    print(f"\n✓ Generated {len(COMPLEX_QUESTIONS)} complex multi-step questions")
    print("\nQuestions:")
    for i, q_dict in enumerate(COMPLEX_QUESTIONS, 1):
        print(f"\n  Q{i}: {q_dict['question']}")
        print(f"      Complexity: {q_dict['complexity'][:60]}...")
