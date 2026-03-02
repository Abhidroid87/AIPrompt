"""
Knowledge Graph module: Store and manage extracted facts as a simple dictionary-based KG.
"""

from typing import Dict, List, Tuple, Optional
from utils import call_ollama, parse_triples


class KnowledgeGraph:
    """
    Simple dictionary-based knowledge graph.
    Structure: {entity: {"facts": [...], "sources": [...], "confidence": float}}
    """
    
    def __init__(self):
        """Initialize empty knowledge graph."""
        self.kg: Dict[str, Dict] = {}
    
    def add_facts(self, answer: str, retrieved_docs: List[str]) -> Dict:
        """
        Extract triples from answer and retrieved docs, add to KG.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved context documents
        
        Returns:
            Dictionary with extracted triples and added entities
        """
        context = "\n".join(retrieved_docs)
        
        # Prompt to extract triples
        prompt = f"""Extract subject-predicate-object triples from this text:

Text:
{answer}

Context:
{context}

List triples in format: "subject | predicate | object" (one per line).
Extract at least 3-5 key facts."""
        
        response = call_ollama(prompt, temperature=0.3)
        
        if not response:
            return {"triples": [], "new_entities": 0}
        
        # Parse triples
        triples = self._parse_triples_from_response(response)
        
        new_entities = 0
        
        # Add to KG
        for subject, predicate, obj in triples:
            subject = subject.strip().lower()
            obj = obj.strip().lower()
            
            if subject not in self.kg:
                self.kg[subject] = {
                    "facts": [],
                    "sources": [],
                    "confidence": 0.0
                }
                new_entities += 1
            
            fact = f"{predicate} {obj}"
            if fact not in self.kg[subject]["facts"]:
                self.kg[subject]["facts"].append(fact)
                self.kg[subject]["sources"].append("answer")
                self.kg[subject]["confidence"] = min(1.0, 
                    self.kg[subject]["confidence"] + 0.3)
        
        return {
            "triples": triples,
            "new_entities": new_entities,
            "total_entities": len(self.kg)
        }
    
    def _parse_triples_from_response(self, response: str) -> List[Tuple[str, str, str]]:
        """
        Parse triples from LLM response.
        Expects format: "subject | predicate | object" per line.
        
        Args:
            response: LLM response
        
        Returns:
            List of (subject, predicate, object) tuples
        """
        triples = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or '|' not in line:
                continue
            
            try:
                parts = line.split('|')
                if len(parts) >= 3:
                    subject = parts[0].strip()
                    predicate = parts[1].strip()
                    obj = parts[2].strip()
                    
                    if subject and predicate and obj:
                        triples.append((subject, predicate, obj))
            except:
                continue
        
        return triples
    
    def query_entity(self, entity: str) -> Optional[Dict]:
        """
        Query KG for entity information.
        
        Args:
            entity: Entity to query
        
        Returns:
            Entity data or None if not found
        """
        entity = entity.strip().lower()
        return self.kg.get(entity, None)
    
    def get_stats(self) -> Dict:
        """
        Get knowledge graph statistics.
        
        Returns:
            Dictionary with KG stats
        """
        total_facts = sum(len(data["facts"]) for data in self.kg.values())
        avg_confidence = (sum(data["confidence"] for data in self.kg.values()) 
                         / len(self.kg)) if self.kg else 0.0
        
        return {
            "entities": len(self.kg),
            "total_facts": total_facts,
            "avg_confidence": avg_confidence
        }
