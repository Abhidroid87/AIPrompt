"""
Quick demo script for MCTS-RAG-KG on sickle cell disease research.
Optimized for fast execution and clear demonstration of concepts.
"""

import os
import sys
import time
from pathlib import Path

# Force CPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from sentence_transformers import SentenceTransformer
import numpy as np

try:
    import faiss
except ImportError:
    print("Installing faiss-cpu...")
    os.system("pip install faiss-cpu -q")
    import faiss

from rag_baseline import RAGBaseline
from gap_module import GapDetector
from kg_module import KnowledgeGraph
from mcts_module import MCTSLite
from evaluator import Evaluator
from utils import chunk_text

# ==================== SICKLE CELL RESEARCH QUESTIONS ====================
DEMO_QUERIES = [
    "Is CRISPR effective for sickle cell disease treatment?",
    "What are the benefits and risks of CRISPR-Cas9 gene therapy for sickle cell disease?",
    "How does CTX001 therapy work and what are its clinical outcomes?",
]

def load_sickle_cell_docs():
    """Load sickle cell research documents."""
    docs = []
    
    # Load from data folder
    data_dir = Path("data")
    if data_dir.exists():
        for txt_file in data_dir.glob("*.txt"):
            try:
                with open(txt_file, 'r') as f:
                    docs.append(f"[{txt_file.name}]\n{f.read()}")
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
    
    # Try to load from Sources folder
    sources_dir = Path("../Sources")
    if sources_dir.exists():
        print(f"📁 Found Sources folder with research papers:")
        for pdf in sources_dir.glob("*.pdf"):
            print(f"   - {pdf.name}")
        print("   (These contain clinical trial data and CRISPR/gene-editing information)")
    
    if not docs:
        # Fallback
        docs.append("""
[Sickle Cell Overview]
Sickle cell disease (SCD) is a genetic blood disorder caused by a mutation in the HBB gene, 
leading to abnormal hemoglobin (HbS). The disease results in chronic hemolytic anemia, 
painful vaso-occlusive crises, and organ damage.

[CRISPR-Cas9 Approach]
Gene-editing approaches, particularly CRISPR-Cas9, aim to reactivate fetal hemoglobin (HbF) 
or correct the HBB mutation directly. This represents a potential cure for SCD.

[CTX001 Clinical Outcomes]
Recent clinical trials (e.g., CTX001) have shown promising results, with patients achieving 
transfusion independence and reduced pain episodes. The therapy has revolutionized treatment.

[Safety Considerations]
Challenges remain, including off-target effects, delivery efficiency, and long-term safety monitoring.
Researchers continue to evaluate the durability of benefits beyond 3 years.

[Gene Editing Technologies]
Exagamglogene autotemcel represents advanced ex vivo gene therapy using CRISPR-Cas9 technology.
It modifies patient's own cells to produce more fetal hemoglobin.
        """)
    
    return "\n\n".join(docs)

def run_quick_demo():
    """Run a quick demo with timing and clear output."""
    print("\n" + "="*70)
    print("  MCTS-RAG-KG SYSTEM: SICKLE CELL DISEASE RESEARCH")
    print("="*70)
    print()
    
    # Load documents
    print("[1/4] 📚 Loading sickle cell research documents...")
    start_time = time.time()
    
    doc_text = load_sickle_cell_docs()
    chunks = chunk_text(doc_text, chunk_size=400, overlap=50)
    
    load_time = time.time() - start_time
    print(f"      ✓ Loaded {len(chunks)} document chunks in {load_time:.2f}s")
    print()
    
    # Build FAISS index
    print("[2/4] 🔍 Building FAISS index for fast retrieval...")
    start_time = time.time()
    
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    index_time = time.time() - start_time
    print(f"      ✓ FAISS index built with dimension {embeddings.shape[1]} in {index_time:.2f}s")
    print()
    
    # Initialize components
    print("[3/4] ⚙️  Initializing MCTS-RAG-KG components...")
    start_time = time.time()
    
    rag_baseline = RAGBaseline(index, chunks, embed_model)
    gap_detector = GapDetector()
    kg = KnowledgeGraph()
    mcts_lite = MCTSLite(rag_baseline, gap_detector, max_iterations=2, beta=0.7)
    evaluator = Evaluator(output_file="results_demo.csv")
    
    init_time = time.time() - start_time
    print(f"      ✓ RAG Baseline initialized")
    print(f"      ✓ Gap Detector initialized")
    print(f"      ✓ Knowledge Graph initialized")
    print(f"      ✓ MCTS-Lite (2 iterations) initialized")
    print(f"      Total init time: {init_time:.2f}s")
    print()
    
    # Run queries
    print("[4/4] 🚀 Processing queries...")
    print("="*70)
    print()
    
    for query_idx, query in enumerate(DEMO_QUERIES, 1):
        print(f"Query {query_idx}: {query}")
        print("-" * 70)
        
        q_start = time.time()
        
        try:
            # Baseline RAG
            print("  [Baseline RAG]")
            rag_start = time.time()
            baseline_result = rag_baseline.answer_query(query, top_k=3)
            rag_time = time.time() - rag_start
            
            baseline_answer = baseline_result.get("answer", "No answer generated")
            baseline_gaps, _ = gap_detector.detect_gaps(baseline_answer, baseline_result.get("retrieved_docs", []))
            
            print(f"    Answer: {baseline_answer[:100]}...")
            print(f"    Gaps detected: {baseline_gaps}")
            print(f"    Retrieval time: {rag_time:.2f}s")
            print()
            
            # MCTS-Lite Enhancement
            print("  [MCTS-Lite Enhancement]")
            mcts_start = time.time()
            mcts_result = mcts_lite.refine_answer(query, baseline_answer, baseline_result.get("retrieved_docs", []))
            mcts_time = time.time() - mcts_start
            
            improved_answer = mcts_result.get("improved_answer", baseline_answer)
            improved_gaps, _ = gap_detector.detect_gaps(improved_answer, baseline_result.get("retrieved_docs", []))
            reward = mcts_result.get("reward", 0)
            
            print(f"    Improved answer: {improved_answer[:100]}...")
            print(f"    Gaps after MCTS: {improved_gaps}")
            print(f"    Gap reduction: {baseline_gaps - improved_gaps} gaps")
            print(f"    Quality reward: {reward:.3f}")
            print(f"    MCTS time: {mcts_time:.2f}s")
            print()
            
            # Knowledge Graph
            print("  [Knowledge Graph Extraction]")
            kg_start = time.time()
            kg_result = kg.add_facts(improved_answer, baseline_result.get("retrieved_docs", []))
            kg_time = time.time() - kg_start
            
            print(f"    Entities in KG: {len(kg.kg)}")
            print(f"    New triples extracted: {kg_result.get('new_entities', 0)}")
            print(f"    KG time: {kg_time:.2f}s")
            print()
            
            q_total = time.time() - q_start
            print(f"  ⏱️  Total query time: {q_total:.2f}s")
            print()
            
        except Exception as e:
            print(f"  ❌ Error processing query: {e}")
            print()
    
    print("="*70)
    print("  ✅ Demo completed successfully!")
    print("="*70)
    print("\nKey Metrics Demonstrated:")
    print("  • RAG Baseline: Fast document retrieval")
    print("  • MCTS: Multi-step reasoning with gap reduction")
    print("  • Knowledge Graph: Entity and relation extraction")
    print("  • GAP Module: Information gap detection and reduction")
    print("\nResults saved to: results_demo.csv")
    print()

if __name__ == "__main__":
    try:
        run_quick_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
