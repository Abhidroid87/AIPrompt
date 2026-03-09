# Gap-Aware MCTS-RAG-KG: A Low-Resource Benchmark for Medical RAG

This repository provides a highly optimized, low-resource implementation of the **Gap-Aware MCTS-RAG-KG** architecture. It is designed specifically to run locally on mid-range hardware (laptops) without the need for large cloud GPUs or heavy graph databases.

This codebase demonstrates the core logic of the Monte Carlo Tree Search (MCTS) Knowledge Graph (KG) retrieval formulation proposed in our research paper. The benchmark specifically targets resolving information gaps (Conflict, Entity, Temporal, and Coverage gaps) in complex medical domains, specifically using CRISPR-Cas9 trials for Sickle Cell Disease (SCD) as the case study.

## 🚀 Key Innovations Demonstrated

1. **Mathematical UCT-Gap Calculation**: Strict implementation of the gap-aware Upper Confidence Bound for Trees formula: 
   $UCT_{gap}(n_j) = \frac{Q(n_j)}{N(n_j)} + c\sqrt{\frac{\ln N(parent)}{N(n_j)}} + \alpha \cdot GapScore(n_j)$
2. **Local, Low-RAM Knowledge Graph**: Uses NetworkX for zero-overhead graph state persistence rather than heavy databases like Neo4j.
3. **Mid-Range Hardware Compatibility**: Embeddings are computed on CPU (FAISS + MiniLM), and LLM synthesis utilizes a quantized lightweight model via Ollama.
4. **Source Credibility Weighting**: Mathematically resolves conflicts between documents (e.g., Clinical Trials vs. PR Blogs) utilizing statistical source credibility factors ($\sigma$).

---

## 🛠️ Setup Instructions for Reviewers

### 1. Prerequisites
- **Python 3.12+**
- **Ollama**: Installed locally to run the local LLM. (Download from [ollama.com](https://ollama.com))

### 2. Environment Setup

Clone or open the repository, then set up the python virtual environment:

```bash
cd mctsragg/
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Model Initialization
You need to pull the specific local LLM we use for reasoning and synthesis:
```bash
ollama pull qwen3:4b
```
*(Ensure the Ollama service is running in the background: `ollama serve`)*

### 4. Ensure Data Sources are Available
The PDFs utilized to generate the answers must be located in the `Sources/` directory at the root of the project. This includes:
1. `Frangoul H, et al_ NEJM 2024_ Exagamglogene autotemcel for severe sickle cell disease.pdf`
2. `december-8-2023-summary-basis-for-regulatory-action-casgevy.pdf`
3. `Vertex and CRISPR Therapeutics Present New Data...pdf`

---

## 🔬 Running the Benchmark

From within the `mctsragg/` directory, simply execute the main trace script:

```bash
cd mctsragg/
./venv/bin/python main.py
```

### Expected Output Trace

The system will output a definitive trace corresponding to our research paper's methodology:
1. **[Retriever]**: FAISS chunking and local index building.
2. **[Planner]**: Entity extraction into the networkx Graph.
3. **[Validator]**: Identifies the **Conflict Gap** regarding efficacy (95% vs. 88%).
4. **[MCTS UCT Trace]**: Evaluates the action and calculates the UCT reward mathematically. 
5. **[Resolution]**: The mathematical resolution where the 95% efficacy is weighted stronger due to clinical source credibility ($\sigma = 0.95$).
6. **[Final Gap Report]**: Formally detects the **Coverage Gap** (lack of >5 year longitudinal data).
7. **[Final Generation]**: Local LLM synthesis of purely the validated factual graph, avoiding hallucinations.

---

## 📁 Repository Structure

* `ARCHITECTURE.md` - In-depth breakdown of the logical workflow and MCTS formulas.
* `Sources/` - The clinical trial PDFs acting as the Knowledge Base.
* `mctsragg/` - The source code.
  * `main.py` - Core MCTS-RAG logic, UCT calculations, NetworkX KG, and Execution Trace.
  * `pdf_parser.py` - Safely extracts chunked texts from papers without memory leaks.
  * `utils.py` - Standardized calls linking to the local Ollama instance and chunking maths.
