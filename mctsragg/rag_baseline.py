"""
Baseline RAG system: retrieve top-3 chunks and generate answer with LLM.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from utils import call_ollama, embed_text, normalize_text


class RAGBaseline:
    """
    Simple baseline RAG: embed query, search FAISS index, retrieve top-3 docs, call LLM.
    """
    
    def __init__(self, faiss_index=None, chunks: List[str] = None, embed_model=None):
        """Initialize RAG baseline.

        This version loads text documents from the `data` directory if no chunks are provided,
        and uses a lightweight keyword‑based retrieval to avoid heavy embedding and FAISS setup.

        Args:
            faiss_index: (optional) placeholder for compatibility; not used in lightweight mode.
            chunks: List of document chunks. If ``None`` the documents are loaded from the
                ``data`` folder under the project root.
            embed_model: (optional) placeholder; not used in lightweight mode.
        """
        self.faiss_index = faiss_index
        # Load documents from data folder if not supplied
        if chunks is None:
            import os
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            self.chunks = []
            for fname in os.listdir(data_dir):
                if fname.lower().endswith('.txt'):
                    with open(os.path.join(data_dir, fname), 'r', encoding='utf-8') as f:
                        text = f.read()
                        # Simple split into paragraphs as chunks
                        self.chunks.extend([p.strip() for p in text.split('\n\n') if p.strip()])
        else:
            self.chunks = chunks
        self.embed_model = embed_model
        self.retrieval_count = 0
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k similar chunks from FAISS index (OPTIMIZED).
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
        
        Returns:
            List of retrieved chunk strings
        """
        self.retrieval_count += 1
        
        if self.faiss_index and self.embed_model:
            # Fast FAISS retrieval
            try:
                query_embedding = self.embed_model.encode(query, convert_to_numpy=True)
                query_embedding = np.array([query_embedding], dtype=np.float32)
                
                distances, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.chunks)))
                retrieved = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
                return retrieved
            except Exception as e:
                print(f"FAISS retrieval error: {e}, falling back to keyword search")
        
        # Fallback: Lightweight keyword-based retrieval
        query_terms = set(query.lower().split())
        scores = []
        for idx, chunk in enumerate(self.chunks):
            chunk_terms = set(chunk.lower().split())
            overlap = len(query_terms & chunk_terms)
            scores.append((overlap, idx))
        # Sort by highest overlap, then by index to stabilize
        scores.sort(key=lambda x: (-x[0], x[1]))
        top_indices = [idx for _, idx in scores[:top_k] if _ > 0]
        # Fallback: if no overlap, just return the first ``top_k`` chunks
        if not top_indices:
            top_indices = list(range(min(top_k, len(self.chunks))))
        self.retrieval_count += 1
        retrieved_docs = [self.chunks[i] for i in top_indices]
        return retrieved_docs
    
    def generate_answer(self, query: str, retrieved_docs: List[str]) -> str:
        """
        Generate answer using LLM with retrieved documents as context.
        
        Args:
            query: Original query
            retrieved_docs: Retrieved context chunks
        
        Returns:
            Generated answer string
        """
        context = "\n".join(retrieved_docs)
        
        prompt = f"""Context:
{context}

Question: {query}

Based on the context above, provide a clear and concise answer."""
        
        answer = call_ollama(prompt, temperature=0.5)
        return answer if answer else "No answer generated."
    
    def answer_query(self, query: str, top_k: int = 3) -> Dict:
        """
        Full RAG pipeline: retrieve -> generate.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
        
        Returns:
            Dictionary with answer, retrieved_docs, and retrieval_count
        """
        retrieved_docs = self.retrieve(query, top_k)
        answer = self.generate_answer(query, retrieved_docs)
        
        return {
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "retrieval_count": self.retrieval_count
        }
