"""
Main entry point for MCTSRAG-KG research prototype.

Workflow:
1. Load and chunk documents
2. Build FAISS index with sentence-transformer embeddings
3. For each test query:
   - Run baseline RAG
   - Run MCTS-lite + KG system
   - Evaluate and log results
4. Print summary statistics
"""
import os
import torch

# --- SAFETY VALVE: PREVENT SYSTEM FREEZE ---
# Forces the script to use your 16GB System RAM, not the 4GB VRAM
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

print(f"✅ Stability Check: MCTS logic is running on {device}")
# -------------------------------------------

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    print("Error: FAISS not installed. Run: pip install faiss-cpu")
    exit(1)

from rag_baseline import RAGBaseline
from gap_module import GapDetector
from kg_module import KnowledgeGraph
from mcts_module import MCTSLite
from evaluator import Evaluator
from utils import chunk_text, normalize_text


# Test queries - 20 hard multi-step questions
TEST_QUERIES = [
    "What are the main differences between supervised and unsupervised learning?",
    "How do neural networks use backpropagation to train weights?",
    "Explain the relationship between entropy and information gain in decision trees.",
    "What are the trade-offs between bias and variance in machine learning models?",
    "How does cross-validation help prevent overfitting?",
    "Describe the process of feature engineering and its importance.",
    "What is the difference between classification and regression tasks?",
    "Explain how k-means clustering works and its limitations.",
    "What are ensemble methods and why do they improve model performance?",
    "How does regularization (L1/L2) help prevent overfitting?",
    "Describe the difference between parametric and non-parametric models.",
    "What is the role of activation functions in neural networks?",
    "Explain the concept of transfer learning and its applications.",
    "How do you handle imbalanced datasets in classification tasks?",
    "What is the difference between batch gradient descent and stochastic gradient descent?",
    "Describe the architecture and training process of autoencoders.",
    "What are the challenges in building effective recommendation systems?",
    "How does dimensionality reduction improve model performance?",
    "Explain the concept of attention mechanisms in transformer models.",
    "What are the key differences between RNNs and CNNs?",
]


def load_documents(filepath: str) -> str:
    """Load documents from file."""
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using sample text.")
        return SAMPLE_DOCUMENTS


def build_faiss_index(chunks: list, embed_model) -> faiss.IndexFlatL2:
    """
    Build FAISS index from document chunks.

    Args:
        chunks: List of text chunks
        embed_model: Sentence transformer model

    Returns:
        FAISS index
    """
    print("Building embeddings...")
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Create and populate FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"FAISS index built with {len(chunks)} chunks, dimension {dimension}")
    return index


def main():
    """Main execution pipeline."""
    print("="*80)
    print("MCTSRAG-KG: Research Prototype")
    print("Baseline RAG vs Gap-Aware MCTS-lite + KG")
    print("="*80)

    # ==================== Setup ====================
    print("\n[1] Loading components...")

    # Load embedding model
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedding model loaded")

    # Load and chunk documents
    doc_text = load_documents("data/documents.txt")
    chunks = chunk_text(doc_text, chunk_size=300, overlap=50)
    print(f"✓ Loaded {len(chunks)} document chunks")

    # Build FAISS index
    faiss_index = build_faiss_index(chunks, embed_model)
    print("✓ FAISS index built")

    # Initialize systems
    rag_baseline = RAGBaseline(faiss_index, chunks, embed_model)
    gap_detector = GapDetector()
    kg = KnowledgeGraph()
    mcts_lite = MCTSLite(rag_baseline, gap_detector, max_iterations=2)
    evaluator = Evaluator(output_file="results.csv")

    print("✓ All components initialized\n")

    # ==================== Evaluation ====================
    print("[2] Running evaluation on test queries...")
    print(f"Testing on {len(TEST_QUERIES)} queries\n")

    for i, query in enumerate(tqdm(TEST_QUERIES, desc="Evaluating"), 1):
        try:
            # Run baseline
            baseline_result = rag_baseline.answer_query(query, top_k=3)

            # Run MCTS-lite
            mcts_result = mcts_lite.refine_answer(
                query,
                baseline_result["answer"],
                baseline_result["retrieved_docs"]
            )

            # Add to KG
            kg_info = kg.add_facts(mcts_result["improved_answer"],
                                   baseline_result["retrieved_docs"])

            # Evaluate
            comparison = evaluator.compare_answers(
                query,
                baseline_result,
                mcts_result,
                gap_detector
            )

            evaluator.log_result(comparison)

            # Print sample result
            if i == 1:
                print(f"\n[Sample Result - Query 1]")
                print(f"Query: {query}")
                print(f"\nBaseline Answer:\n  {baseline_result['answer'][:200]}...")
                print(f"\nMCTS Answer:\n  {mcts_result['improved_answer'][:200]}...")
                print(f"\nImprovement: {comparison['improvement_score']:.2%}\n")

        except Exception as e:
            print(f"Error processing query {i}: {e}")
            continue

    # ==================== Summary ====================
    print("\n[3] Evaluation complete!\n")

    # Print summary
    evaluator.print_summary()

    # Print KG stats
    kg_stats = kg.get_stats()
    print(f"Knowledge Graph Statistics:")
    print(f"  Entities extracted: {kg_stats['entities']}")
    print(f"  Total facts: {kg_stats['total_facts']}")
    print(f"  Average confidence: {kg_stats['avg_confidence']:.2f}\n")

    print(f"Results saved to: results.csv")


# Sample documents for testing if data/documents.txt is not found
SAMPLE_DOCUMENTS = """
Machine Learning Overview:

Supervised Learning is a category of machine learning that uses labeled training data.
In supervised learning, each training example includes input features and the target output.
The algorithm learns to map inputs to outputs by minimizing prediction error.
Common supervised learning tasks include classification and regression.

Unsupervised Learning operates on unlabeled data without predefined target outputs.
The goal is to discover hidden patterns or structures in the data.
Clustering algorithms like k-means group similar data points together.
Dimensionality reduction techniques like PCA reduce data complexity.

Neural Networks are computational models inspired by biological neurons.
They consist of interconnected layers of nodes or "neurons."
Backpropagation is the primary algorithm used to train neural networks.
During backpropagation, gradients flow backwards through the network to update weights.

Decision Trees recursively split data based on feature values.
At each node, the algorithm selects the feature that provides the best information gain.
Entropy measures the disorder or impurity in a dataset.
Information gain measures the reduction in entropy after a split.

Overfitting occurs when a model learns training data too well and fails to generalize.
High model complexity relative to training data causes overfitting.
Cross-validation estimates generalization error by training on multiple data splits.
Regularization techniques like L1 and L2 penalize large weights to reduce overfitting.

Feature engineering involves creating new features from raw data.
Good features improve model performance and interpretability.
Feature scaling normalizes features to a similar range.
Feature selection identifies the most relevant features for prediction.

Ensemble Methods combine multiple models to improve performance.
Bagging trains models on random subsets of data with replacement.
Boosting trains models sequentially, focusing on previously misclassified examples.
Random Forests combine multiple decision trees through bagging.

Classification assigns data points to discrete categories.
Regression predicts continuous numerical values.
K-means clustering partitions data into k groups based on similarity.
Hierarchical clustering builds a tree of nested clusters.

Activation functions introduce non-linearity to neural networks.
ReLU is a popular activation function that outputs max(0, x).
Sigmoid outputs values between 0 and 1 for binary classification.
Softmax generalizes sigmoid for multi-class classification.

Transfer Learning reuses models trained on large datasets.
Fine-tuning adapts pre-trained models to new tasks with limited data.
Pre-trained embeddings capture semantic relationships between words.
Vision Transformers use transfer learning on image data.
"""


if __name__ == "__main__":
    main()
