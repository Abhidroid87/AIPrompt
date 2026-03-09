# Gap-Aware MCTS-RAG-KG: Technical Architecture

This document details the architectural components, internal workflow, and mathematical formulations of the **MCTS-RAG-KG Pipeline** designed for local, resource-constrained environments.

The architecture directly mitigates the challenge of hallucination and unverified generation in medical LLMs by requiring mathematically validated, Graph-based states before synthesis.

---

## 🏗️ 1. Conceptual Modules

The system is decomposed into four distinct yet interlocking modules to ensure low RAM consumption on mid-range laptops:

### A. Local Retriever (FAISS + MiniLM)
Due to memory constraints, the system **never** passes an entire PDF into the LLM context window. Instead:
- Documents (the 3 Sickle Cell Disease clinical papers/reports) are extracted strictly to text using `pypdfium2`.
- Text is dynamically chunked into blocks of ~250 words.
- Chunks are vectorized into a flat space using `sentence-transformers/all-MiniLM-L6-v2`.
- Vector similarity is processed via CPU-optimized `faiss.IndexFlatL2`.

### B. Minimal Knowledge Graph (NetworkX)
Heavy external graph databases (like Neo4j) are discarded in favor of pure Python dictionaries representing nodes and edges via **NetworkX**.
- **`EntityNode`**: Core concepts required for the prompt (e.g., "CRISPR", "SCD").
- **`FactNode`**: Formatted statements directly retrieved from source papers. Includes a `credibility_score` ($\sigma$).
- **`GapNode`**: Structural omissions detected during generation.

### C. MCTS Planner
Operates iteratively. Nodes in the tree represent a snapshot of the current Graph state. Actions are the targeted retrievals meant to resolve explicit `GapNode`s.

### D. Generator (Ollama + Qwen3.4B)
A single heavily-prompted API call made only at the very end of the execution trace to synthesize the strictly validated factual Graph into natural language.

---

## 🧮 2. The Mathematical Formulation

The core innovation is the modified Upper Confidence Bound for Trees ($UCT_{gap}$) formula used by the MCTS Planner.

### Expanding the Gap Score Calculation
The gap reduction reward is critical for pushing the model towards factual stability.
If $G_{gap}$ refers to the current graph of unresolved gaps, then the action $n$ achieves a reward defined by:

$$GapScore(n) = \sum_{g \in G_{gap}} w(g_{type}) \cdot priority(g) \cdot resolved(g,n)$$

- **$w(g_{type})$**: The structural weight of the gap.
  - Conflict Gaps = 0.9 (Highest penalty, immediate resolution needed)
  - Entity Gaps = 0.7 
  - Coverage Gaps = 0.5 (Missing long-term data)
- **$priority(g)$**: Assigned during graph generation based on the query constraint (0.0 - 1.0).
- **$resolved(g, n)$**: A binary (0 or 1) indicating if the node expansion successfully neutralizes the gap.

### The $UCT_{gap}$ Execution
During MCTS selection, the algorithm calculates the score using:

$$UCT_{gap}(n_j) = \frac{Q(n_j)}{N(n_j)} + c\sqrt{\frac{\ln N(parent)}{N(n_j)}} + \alpha \cdot GapScore(n_j)$$

*Where:*
- $Q(n_j)$: Exploitation value of the generated answer quality.
- $N(n_j)$: Total structural visits to the node.
- $c$: Exploration constant (Standard $1.414$).
- $\alpha$: Weight scalar for the GapScore component.

---

## 🔄 3. System Workflow (Execution Trace)

When running `main.py` against the query:
*"Is CRISPR effective for sickle cell disease?"*

1. **Initialization**: Sources loaded, FAISS vector index generated.
2. **Planner Extraction**: Entities `E1(CRISPR)` and `E2(Sickle Cell Disease)` are pushed to the Graph.
3. **Retrieval Simulation**: LocalRetriever fetches corresponding chunks across the mixed-credibility documents.
4. **Validation/Conflict Detection**: 
   - PDF 1 (NEJM, $\sigma = 0.95$) indicates **95% Efficacy**.
   - PDF 2 (Blog, $\sigma = 0.40$) indicates **88% Efficacy**.
   - **G1 (Conflict Gap)** and **G2 (Coverage Gap: temporal)** are explicitly injected into the $G_{gap}$ set.
5. **MCTS Algorithm Evaluates**: The *UCT-Gap* mathematically triggers an action to prioritize resolving the conflict over expanding new random information.
6. **Resolution Phase**: The Conflict is settled by evaluating the specific chunk's stored `credibility` bounds. The $0.95$ credibility metric of the NEJM forces the Graph to discard the $88\%$ artifact.
7. **Final Synthesis**: The QA engine takes the resolved KG matrix (95% efficacy validated; >5 years durability missing) and feeds it to `Qwen3.4B` for standard summarization. No hallucination occurs because the source content was already mathematically filtered.
