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
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved context documents
        
        Returns:
            Tuple of (gap_count: int, gap_description: str)
        """
        context = "\n".join(retrieved_docs)
        
        prompt = f"""Analyze this answer and identify information gaps:

Context Documents:
{context}

Generated Answer:
{answer}

For the answer, identify and count:
1. Missing entities (proper nouns, key concepts not mentioned)
2. Missing relations (connections between entities not explained)
3. Temporal information gaps (dates, sequences missing)
4. Conflicts or contradictions with the context

Provide a brief summary of gaps found and count the total gaps.
Format: "Gaps found: [NUMBER]. Issues: [brief description]" """
        
        response = call_ollama(prompt, temperature=0.3)
        
        if not response:
            return 0, "Analysis failed"
        
        # Simple extraction of gap count from response
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
