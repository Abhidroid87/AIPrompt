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
    
    def __init__(self, faiss_index, chunks: List[str], embed_model):
        """
        Initialize RAG baseline.
        
        Args:
            faiss_index: FAISS index for similarity search
            chunks: List of document chunks
            embed_model: Sentence transformer model for embeddings
        """
        self.faiss_index = faiss_index
        self.chunks = chunks
        self.embed_model = embed_model
        self.retrieval_count = 0
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve top-k similar chunks from FAISS index.
        
        Args:
            query: Query string
            top_k: Number of chunks to retrieve
        
        Returns:
            List of retrieved chunk strings
        """
        # Embed query
        query_embedding = embed_text(query, self.embed_model)
        if query_embedding is None:
            return []
        
        query_vec = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS
        distances, indices = self.faiss_index.search(query_vec, top_k)
        
        self.retrieval_count += 1
        retrieved_docs = [self.chunks[idx] for idx in indices[0] if idx < len(self.chunks)]
        
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
