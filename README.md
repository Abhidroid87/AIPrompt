# MCTSRAG-KG: Fact-Oriented Data Retrieval via MCTS-Guided Multi-Agent Reasoning

This repository serves as the official proof-of-concept implementation for the research paper: 
**"Fact-Oriented Data Retrieval via MCTS-Guided Multi-Agent Reasoning with Persistent Knowledge Graphs and Systematic Gap Analysis"**

## Overview

MCTSRAG-KG addresses the core limitations of traditional, single-pass Retrieval-Augmented Generation (RAG) systems. Standard RAG assumes retrieved knowledge is self-consistent and complete, leading to unhandled ambiguity and hallucinations. 

This implementation demonstrates an end-to-end framework formulating information retrieval as a **cost-aware planning problem with epistemic uncertainty**. 

### 1. Gap-Prioritized MCTS Planning
As outlined in **Algorithm 1** of our paper, this codebase shifts from a "greedy" linear retrieval loop to an actual **Monte Carlo Tree Search (MCTS)**.
- **Implemented in:** `mcts_module.py`
- **How it works:** Our MCTS constructs a search tree where each node represents a retrieval state (`answer`, `retrieved_docs`). Traversal is explicitly governed by our proposed Gap-aware UCT formula:
  `UCT = Q(n)/N(n) + c * sqrt(ln(N(parent)) / N(n)) + GapScore(n)`
- **Reward Function:** The reward explicitly prioritizes closing epistemic gaps over simple semantic similarity: `Reward = β * ΔGap + (1-β) * Quality`.

### 2. Multi-Agent Reasoning & Coordination
Our paper introduces a coordinated pipeline of a Planner, Retriever, Validator, and Synthesizer. In the codebase, these agents directly map to the **Expansion Phase** of our MCTS algorithm (`mcts_module.py`):
- **Planner/Retriever (`refine_query`):** Generates follow-up questions to resolve missing context or ambiguities.
- **Synthesizer (`expand_entity`):** Expands definitions, attributes, and relationships using prior context.
- **Validator (`verify_answer`):** Acts as a critical self-check against the fetched context to identify contradictions or unsupported facts before formulating the final answer.

### 3. Persistent Knowledge Graph (KG) Memory
Unlike stateless RAGs, our system updates a persistent Knowledge Graph during retrieval. 
- **Implemented in:** `kg_module.py`
- **How it works:** Extracted facts are parsed into Subject-Predicate-Object triples, assigned confidence scores, and aggregated over the session. This provides transparent tracking, attribution to original sources, and allows the framework to detect and flag "Conflict Gaps."

### 4. Epistemic Gap Detector
A tailored LLM sequence explicitly evaluates 5 types of gaps mapping to our framework's taxonomy: Entity, Relation, Temporal, Conflict, and Coverage gaps.
- **Implemented in:** `gap_module.py`
- **How it works:** Evaluates the generated answer against the context, extracting explicitly missing links or contradictory statements. The total gap count translates into the `GapReduction` metric fed to the MCTS Planner.

---

## Experimental Validation & Evidence of Claims

Executing this codebase (`python3 main.py`) empirically validates the claims made in Section 5 of our paper. The output includes an **Evaluation Summary** logging the comparison between Baseline RAG and MCTSRAG-KG across high-complexity, multi-step queries.

**What this code proves regarding our paper claims:**
1. **Factuality & Interpretability:** The final answer is no longer a hidden black-box generation. The MCTS module tracks `actions_log` across nodes, providing complete **Reasoning Traces**. The `kg_module.py` provides exact triples verifying *where* the information originated. 
2. **Reduced Halucinations via Gap Reduction:** By strictly factoring `ΔGap` into the MCTS reward loop (`mcts_module.py`), the final retrieved nodes demonstrably drop the "Open Gap Count" compared to baseline. 
3. **Cost Sensitivity:** Not every query requires full tree rollouts. The MCTS loop dynamically searches path trees, maximizing utility within bounded iterations. 

---

## Getting Started

### Prerequisites

We recommend utilizing an environment configured for local inference (e.g., Ollama) combined with FAISS for semantic routing, reducing API overhead for intense tree-search rollouts.

- CPU / CUDA configured environments (Note: VRAM constrained environments can safely run this logic on System RAM).
- Local LLM Runner (e.g., Ollama) pulling `qwen3:4b` or a similar highly-capable parameter model.

### Execution

```bash
# 1. Install dependencies
pip install sentence-transformers faiss-cpu tqdm requests

# 2. Pull local model for multi-agent logic
ollama pull qwen3:4b

# 3. Execute the pipeline
python3 main.py
```

Results are saved to `results.csv` tracking:
- Faithfulness evaluation
- Iteration cost
- Gap Reduction scores
- Base vs. MCTS improvement metrics

## Future Work & Next Steps
As proposed in the paper, extending this implementation to support **Tree-of-Thoughts** inside agent nodes, as well as enabling **cross-user persistent decentralized KGs**, stand as prime candidates for Version 2.0 of this repository.
