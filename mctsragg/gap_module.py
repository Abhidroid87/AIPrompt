"""
Gap detection module: Identify missing information in answers using LLM.
"""

from typing import Dict, Tuple
from utils import call_ollama


class GapDetector:
    """
    Detect gaps (missing entities, relations, temporal info, conflicts) in answers.
    """
    
    def detect_gaps(self, answer: str, retrieved_docs: list) -> Tuple[int, str]:
        """
        Use LLM to identify gaps in answer based on retrieved documents.
        OPTIMIZED: Faster gap detection for demo.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved context documents
        
        Returns:
            Tuple of (gap_count: int, gap_description: str)
        """
        context = "\n".join(retrieved_docs[:2])  # Use only first 2 docs for speed
        
        prompt = f"""Quickly analyze this answer for information gaps:

Context:
{context}

Answer:
{answer}

Count these gaps:
1. Missing key concepts
2. Unexplained connections
3. Contradictions

Format: "[NUMBER] gaps: [brief list]" """
        
        response = call_ollama(prompt, temperature=0.2)
        
        if not response:
            return 0, "Analysis failed"
        
        # Fast extraction of gap count
        gap_count = 0
        try:
            # Look for "Gaps found: N" pattern
            if "Gaps found:" in response:
                parts = response.split("Gaps found:")[1].split(".")[0].strip()
                gap_count = int(''.join(filter(str.isdigit, parts.split()[0])))
            else:
                # Fallback: count issue mentions
                response_lower = response.lower()
                gap_count = (response_lower.count("missing") + 
                            response_lower.count("gap") + 
                            response_lower.count("not mentioned"))
        except:
            gap_count = max(1, len(response.split("•")) - 1)
        
        return gap_count, response
    
    def get_gap_reduction(self, before_answer: str, after_answer: str, 
                         retrieved_docs: list) -> float:
        """
        Compute gap reduction ratio.
        
        Args:
            before_answer: Answer before refinement
            after_answer: Answer after refinement
            retrieved_docs: Context documents
        
        Returns:
            Gap reduction ratio (0-1)
        """
        gaps_before, _ = self.detect_gaps(before_answer, retrieved_docs)
        gaps_after, _ = self.detect_gaps(after_answer, retrieved_docs)
        
        if gaps_before == 0:
            return 0.0
        
        reduction = (gaps_before - gaps_after) / gaps_before
        return min(max(reduction, 0.0), 1.0)  # Clamp to [0, 1]
