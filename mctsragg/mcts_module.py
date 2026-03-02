"""
MCTS-lite module: Monte-Carlo Tree Search for answer refinement.
Implements gap-aware UCT action selection with reward computation
as defined in the fact-oriented data retrieval paper.
"""

import math
from typing import Dict, List, Tuple, Optional
from utils import call_ollama, embed_text, normalize_text
import numpy as np

class MCTSNode:
    """
    Represents a state in the Monte Carlo Tree Search.
    State consists of the current query, answer, and retrieved documents.
    """
    def __init__(self, answer: str, retrieved_docs: List[str], parent=None, action_taken=None, query=None):
        self.answer = answer
        self.retrieved_docs = retrieved_docs
        self.parent = parent
        self.children = []
        self.visits = 0
        self.reward_sum = 0.0
        self.action_taken = action_taken
        self.query = query
        self.gap_score = 0.0
        
    def expand(self, children_nodes):
        self.children.extend(children_nodes)
        
    def is_leaf(self):
        return len(self.children) == 0

    def uct_score(self, exploration_constant=1.414):
        """
        UCT = Q(n_j)/N(n_j) + c * sqrt(ln(N(n_parent)) / N(n_j)) + GapScore(n_j)
        """
        if self.visits == 0:
            return float('inf')
        
        exploit = self.reward_sum / self.visits
        explore = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits) if self.parent else 0
        
        # Incorporate gap reduction directly into node selection as per paper
        return exploit + explore + self.gap_score

class MCTSLite:
    """
    Gap-Aware MCTS for RAG answer refinement using Multi-Agent specialized actions.
    Actions: refine_query (Planner/Retriever), expand_entity (Synthesizer), verify_answer (Validator)
    """
    
    def __init__(self, rag_baseline, gap_detector, max_iterations: int = 3, beta: float = 0.6):
        """
        Initialize MCTS.
        
        Args:
            rag_baseline: RAG baseline instance for retrieval
            gap_detector: Gap detector instance
            max_iterations: Maximum refinement iterations
            beta: Weight for gap reduction vs quality in reward (R = beta*ΔGap + (1-beta)*Quality)
        """
        self.rag = rag_baseline
        self.gap_detector = gap_detector
        self.max_iterations = max_iterations
        self.beta = beta
    
    def refine_query(self, original_query: str, answer: str, 
                     retrieved_docs: List[str]) -> str:
        """Action 1: Refine query to fetch more documents, aiming at Planner/Retriever behavior."""
        prompt = f"""The original question: {original_query}

Current answer: {answer}

Based on this answer, generate a follow-up question that:
1. Asks for missing context or details
2. Clarifies any ambiguities
3. Explores related entities

Provide only the refined question."""
        
        refined_query = call_ollama(prompt, temperature=0.7)
        return refined_query if refined_query else original_query
    
    def expand_entity(self, answer: str, retrieved_docs: List[str]) -> str:
        """Action 2: Synthesize and expand details about entities in the answer."""
        context = "\n".join(retrieved_docs)
        
        prompt = f"""Given this answer: {answer}

And this context: {context}

Expand the answer by providing more details about key entities mentioned.
Include: definitions, attributes, relationships to other entities."""
        
        expanded = call_ollama(prompt, temperature=0.5)
        return expanded if expanded else answer
    
    def verify_answer(self, answer: str, retrieved_docs: List[str]) -> str:
        """Action 3: Validator assessing the answer against retrieved documents to fix conflicts."""
        context = "\n".join(retrieved_docs)
        
        prompt = f"""Review this answer: {answer}

Against this context: {context}

Identify any parts of the answer that:
1. Are not supported by the context
2. Are contradicted by the context
3. Need clarification

Provide a corrected version of the answer that is fully supported."""
        
        verified = call_ollama(prompt, temperature=0.3)
        return verified if verified else answer
    
    def compute_quality_score(self, answer: str, retrieved_docs: List[str]) -> float:
        """Estimate answer quality (faithfulness to docs)."""
        # Join documents but limit total context length
        context_text = "\n".join(retrieved_docs)
        if len(context_text) > 4000:
            context_text = context_text[:4000] + "..."
            
        prompt = f"""Evaluate how well the following answer is supported by the provided context.
        
Answer: {answer}

Context Documents:
{context_text}

Rate the faithfulness on a scale 0.0 to 1.0 where:
0.0 = Not supported or contradicted
0.5 = Partially supported
1.0 = Fully supported

Respond ONLY with the numerical score (e.g., 0.8)."""
        
        response = call_ollama(prompt, temperature=0.3)
        
        try:
            cleaned_response = response.strip()
            # Handle cases where model returns non-numeric prefix
            import re
            numeric_match = re.search(r"(\d+\.\d+|\d+)", cleaned_response)
            if numeric_match:
                score = float(numeric_match.group(1))
            else:
                score = 0.5
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5
            
    def _compute_reward(self, answer: str, retrieved_docs: List[str], baseline_gaps: int) -> Tuple[float, float, int]:
        r"""
        Compute reward per paper: R = \beta * \Delta Gap + (1-\beta) * Quality
        Returns (Reward, GapScore, GapCount)
        """
        gap_count, _ = self.gap_detector.detect_gaps(answer, retrieved_docs)
        
        # Calculate Delta Gap Reduction
        gap_reduction = 0.0
        if baseline_gaps > 0:
            gap_reduction = max(0.0, (baseline_gaps - gap_count) / baseline_gaps)
        elif gap_count == 0:
            gap_reduction = 1.0
            
        quality_score = self.compute_quality_score(answer, retrieved_docs)
        
        # Combined reward based on paper formula
        reward = self.beta * gap_reduction + (1 - self.beta) * quality_score
        
        return min(max(reward, 0.0), 1.0), gap_reduction, gap_count

    def refine_answer(self, query: str, initial_answer: str, 
                     retrieved_docs: List[str]) -> Dict:
        """
        MCTS reasoning loop.
        Constructs a tree iteratively evaluating potential trajectories.
        """
        # 1. Initialize Root
        root = MCTSNode(answer=initial_answer, retrieved_docs=retrieved_docs, query=query)
        baseline_gaps, _ = self.gap_detector.detect_gaps(initial_answer, retrieved_docs)
        
        reward_val, gap_score_val, gap_count_val = self._compute_reward(initial_answer, retrieved_docs, baseline_gaps)
        root.reward_sum = reward_val
        root.gap_score = gap_score_val
        root.visits = 1
        
        # Keep track of all docs seen during search
        cumulative_docs = list(retrieved_docs)
        
        actions_log = []
        
        for iteration in range(self.max_iterations):
            # Selection Phase
            current = root
            while not current.is_leaf():
                # Select child with max UCT
                current = max(current.children, key=lambda c: c.uct_score())
                
            # Expansion Phase
            candidate_children = []
            
            # Action 1: Refine Query (Planner/Retriever)
            refined_query = self.refine_query(current.query or query, current.answer, current.retrieved_docs)
            refined_docs = self.rag.retrieve(refined_query, top_k=3)
            all_docs = list(set(current.retrieved_docs + refined_docs))
            refined_answer = self.rag.generate_answer(refined_query, all_docs)
            
            child1 = MCTSNode(answer=refined_answer, retrieved_docs=all_docs, parent=current, action_taken="refine_query", query=refined_query)
            candidate_children.append(child1)
            
            # Action 2: Audit & Resolve (Validator) - Combined Entity Expansion & Verification
            verified_answer = self.verify_answer(current.answer, current.retrieved_docs)
            child2 = MCTSNode(answer=verified_answer, retrieved_docs=current.retrieved_docs, parent=current, action_taken="audit_and_resolve", query=current.query)
            candidate_children.append(child2)
            
            # Simulation & Backpropagation Phase
            for child in candidate_children:
                reward, gap_score, gc = self._compute_reward(child.answer, child.retrieved_docs, baseline_gaps)
                child.reward_sum = reward
                child.gap_score = gap_score
                child.visits = 1
                
                # Update global doc set
                for doc in child.retrieved_docs:
                    if doc not in cumulative_docs:
                        cumulative_docs.append(doc)
                
                # Backpropagate to root
                temp = child.parent
                while temp is not None:
                    temp.visits += 1
                    temp.reward_sum += reward
                    temp = temp.parent
                    
            current.expand(candidate_children)
            
            actions_log.append({
                "iteration": iteration,
                "node_expanded": current.action_taken or "root"
            })
            
        # Select best overall answer path
        # Find the node with the highest average reward derived across the search
        best_child = max(root.children, key=lambda c: c.reward_sum / max(1, c.visits)) if root.children else root
        
        return {
            "improved_answer": best_child.answer,
            "initial_answer": initial_answer,
            "reward": best_child.reward_sum / max(1, best_child.visits),
            "iterations": self.max_iterations,
            "actions": actions_log,
            "all_retrieved_docs": cumulative_docs  # VERY IMPORTANT FOR EVALUATION
        }
