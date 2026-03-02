"""
Utility functions for MCTSRAG-KG system.
"""

import requests
import json
from typing import Optional

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:4b"


def call_ollama(prompt: str, temperature: float = 0.7) -> Optional[str]:
    """
    Call Ollama Qwen 3.4 model via REST API (OPTIMIZED).
    
    Args:
        prompt: The input prompt for the model
        temperature: Sampling temperature (0-1)
    
    Returns:
        Generated text response or None if error
    """
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "temperature": temperature,
                "stream": False,
                "options": {
                    "num_predict": 1024,  # Increased for reasoning models
                    "top_k": 40,
                    "top_p": 0.9,
                    "num_ctx": 4096      # Ensure enough context window
                }
            },
            timeout=300,  # Increased timeout for reasoning/thinking time
        )
        if response.status_code == 200:
            result = response.json()
            full_response = result.get("response", "").strip()
            
            # If the model has a reasoning section (common in Qwen/DeepSeek reasoning models)
            # Try to extract the final answer.
            if "...done thinking." in full_response:
                parts = full_response.split("...done thinking.")
                return parts[-1].strip()
            elif "</thought>" in full_response:
                parts = full_response.split("</thought>")
                return parts[-1].strip()
            
            return full_response
        else:
            print(f"Ollama error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def embed_text(text: str, model) -> Optional[list]:
    """
    Embed text using sentence-transformers model.
    
    Args:
        text: Text to embed
        model: Loaded sentence-transformer model
    
    Returns:
        Embedding vector or None
    """
    try:
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None


def normalize_text(text: str) -> str:
    """Remove extra whitespace and normalize text."""
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    
    return chunks


def parse_triples(text: str) -> list:
    """
    Simple parser for subject-predicate-object triples from LLM output.
    Looks for patterns like "subject predicate object" or "S-P-O".
    
    Args:
        text: Text potentially containing triples
    
    Returns:
        List of (subject, predicate, object) tuples
    """
    triples = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        
        # Simple heuristic: look for lines with 3+ words that might be triples
        parts = line.split()
        if len(parts) >= 3:
            # Take first 3 significant words as S-P-O
            triple = (parts[0], parts[1], " ".join(parts[2:]))
            triples.append(triple)
    
    return triples
