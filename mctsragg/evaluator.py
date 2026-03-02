"""
Evaluation module: Compare baseline vs MCTS-lite answers.
Measure faithfulness, retrieval cost, and gap reduction.
"""

import csv
from datetime import datetime
from typing import Dict, List
from utils import call_ollama


class Evaluator:
    """
    Evaluate and compare RAG baseline vs MCTS-lite system.
    """
    
    def __init__(self, output_file: str = "results.csv"):
        """
        Initialize evaluator.
        
        Args:
            output_file: CSV file to log results
        """
        self.output_file = output_file
        self.results = []
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            "timestamp",
            "query",
            "baseline_answer",
            "mcts_answer",
            "baseline_faithfulness",
            "mcts_faithfulness",
            "baseline_gaps",
            "mcts_gaps",
            "baseline_retrievals",
            "mcts_retrievals",
            "improvement_score",
            "mcts_iterations"
        ]
        
        try:
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            print(f"Error initializing CSV: {e}")
    
    def evaluate_faithfulness(self, answer: str, retrieved_docs: List[str]) -> bool:
        """
        Judge if answer is faithful to retrieved documents using LLM.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved context documents
        
        Returns:
            True if faithful, False otherwise
        """
        context = "\n".join(retrieved_docs)
        
        prompt = f"""Is this answer faithful to the provided documents?

Answer: {answer}

Documents:
{context}

A faithful answer should be supported by the documents.
Respond with only: "yes" or "no" """
        
        response = call_ollama(prompt, temperature=0.1)
        
        if response:
            return response.lower().strip().startswith("yes")
        return False
    
    def compare_answers(self, query: str, baseline_result: Dict, 
                       mcts_result: Dict, gap_detector) -> Dict:
        """
        Compare baseline and MCTS-lite results.
        
        Args:
            query: Original query
            baseline_result: Result from baseline RAG
            mcts_result: Result from MCTS-lite
            gap_detector: Gap detector for analysis
        
        Returns:
            Comparison dictionary with metrics
        """
        baseline_answer = baseline_result["answer"]
        mcts_answer = mcts_result["improved_answer"]
        baseline_docs = baseline_result["retrieved_docs"]
        
        # Faithfulness
        baseline_faithful = self.evaluate_faithfulness(baseline_answer, baseline_docs)
        mcts_faithful = self.evaluate_faithfulness(mcts_answer, baseline_docs)
        
        # Gap count
        baseline_gaps, _ = gap_detector.detect_gaps(baseline_answer, baseline_docs)
        mcts_gaps, _ = gap_detector.detect_gaps(mcts_answer, baseline_docs)
        
        # Retrieval cost
        baseline_retrievals = baseline_result.get("retrieval_count", 1)
        mcts_retrievals = baseline_retrievals + (mcts_result.get("iterations", 0) * 3)
        
        # Improvement score (0-1)
        gap_improvement = 0.0
        if baseline_gaps > 0:
            gap_improvement = (baseline_gaps - mcts_gaps) / baseline_gaps
        
        faithfulness_improvement = 0.0
        if baseline_faithful != mcts_faithful:
            faithfulness_improvement = 1.0 if mcts_faithful else -0.5
        
        overall_improvement = (0.7 * gap_improvement + 0.3 * faithfulness_improvement)
        overall_improvement = min(max(overall_improvement, 0.0), 1.0)
        
        return {
            "query": query,
            "baseline_answer": baseline_answer,
            "mcts_answer": mcts_answer,
            "baseline_faithful": baseline_faithful,
            "mcts_faithful": mcts_faithful,
            "baseline_gaps": baseline_gaps,
            "mcts_gaps": mcts_gaps,
            "baseline_retrievals": baseline_retrievals,
            "mcts_retrievals": mcts_retrievals,
            "improvement_score": overall_improvement,
            "mcts_iterations": mcts_result.get("iterations", 0)
        }
    
    def log_result(self, comparison: Dict):
        """
        Log comparison result to CSV.
        
        Args:
            comparison: Comparison dictionary from compare_answers()
        """
        try:
            with open(self.output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    comparison["query"][:100],
                    comparison["baseline_answer"][:100],
                    comparison["mcts_answer"][:100],
                    comparison["baseline_faithful"],
                    comparison["mcts_faithful"],
                    comparison["baseline_gaps"],
                    comparison["mcts_gaps"],
                    comparison["baseline_retrievals"],
                    comparison["mcts_retrievals"],
                    f"{comparison['improvement_score']:.2f}",
                    comparison["mcts_iterations"]
                ])
        except Exception as e:
            print(f"Error logging result: {e}")
        
        self.results.append(comparison)
    
    def print_summary(self):
        """Print summary statistics of all evaluations."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        total = len(self.results)
        baseline_faithful = sum(1 for r in self.results if r["baseline_faithful"])
        mcts_faithful = sum(1 for r in self.results if r["mcts_faithful"])
        
        avg_baseline_gaps = sum(r["baseline_gaps"] for r in self.results) / total
        avg_mcts_gaps = sum(r["mcts_gaps"] for r in self.results) / total
        
        avg_improvement = sum(r["improvement_score"] for r in self.results) / total
        total_retrievals_baseline = sum(r["baseline_retrievals"] for r in self.results)
        total_retrievals_mcts = sum(r["mcts_retrievals"] for r in self.results)
        
        print(f"Total queries evaluated: {total}")
        print(f"\nFaithfulness:")
        print(f"  Baseline: {baseline_faithful}/{total} ({100*baseline_faithful/total:.1f}%)")
        print(f"  MCTS-lite: {mcts_faithful}/{total} ({100*mcts_faithful/total:.1f}%)")
        print(f"\nAverage gaps per answer:")
        print(f"  Baseline: {avg_baseline_gaps:.2f}")
        print(f"  MCTS-lite: {avg_mcts_gaps:.2f}")
        print(f"  Reduction: {avg_baseline_gaps - avg_mcts_gaps:.2f}")
        print(f"\nImprovement Score (avg): {avg_improvement:.2f}")
        print(f"\nRetrieval Cost:")
        print(f"  Baseline total: {total_retrievals_baseline}")
        print(f"  MCTS-lite total: {total_retrievals_mcts}")
        print(f"  Overhead ratio: {total_retrievals_mcts/total_retrievals_baseline:.2f}x")
        print("="*80 + "\n")
