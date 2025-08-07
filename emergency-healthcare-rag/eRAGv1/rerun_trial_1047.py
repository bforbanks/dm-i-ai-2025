#!/usr/bin/env python3
"""
Rerun evaluation with exact parameters from trial 1047
Evaluate top-1, top-2, and top-3 accuracy
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import from bm25_bayesian_optimization
sys.path.append(str(Path(__file__).parent))

from bm25_bayesian_optimization import evaluate_config, build_bm25_index, bm25_search, load_statements

def evaluate_top_k_accuracy(chunk_size: int, overlap: int, k1: float, b: float, max_k: int = 3):
    """Evaluate BM25 configuration for top-1, top-2, ..., top-k accuracy."""
    statements = load_statements()
    
    # Build index (only once)
    print("🔨 Building BM25 index...")
    data = build_bm25_index(chunk_size, overlap, k1, b)
    
    # Initialize counters for each k
    correct_counts = {k: 0 for k in range(1, max_k + 1)}
    total = len(statements)
    
    print(f"📊 Evaluating {total} statements...")
    
    # Single pass through all statements
    for i, (stmt, true_topic) in enumerate(statements):
        if i % 100 == 0:
            print(f"   Progress: {i}/{total}")
        
        # Get top-3 results in one search
        results = bm25_search(stmt, data, top_k=max_k)
        
        # Check for each k if the true topic is in the top-k results
        for k in range(1, max_k + 1):
            top_k_topics = [r['topic_id'] for r in results[:k]]
            if true_topic in top_k_topics:
                correct_counts[k] += 1
    
    # Calculate accuracies
    accuracies = {k: correct_counts[k] / total for k in range(1, max_k + 1)}
    return accuracies

def main():
    # Parameters from trial 1047
    chunk_size = 101
    overlap = 59
    k1 = 0.7501082129755392
    b = 0.6084497002973362
    
    print("🔄 Rerunning trial 1047 with exact parameters:")
    print(f"   chunk_size: {chunk_size}")
    print(f"   overlap: {overlap}")
    print(f"   k1: {k1}")
    print(f"   b: {b}")
    print("-" * 50)
    
    # Run evaluation for top-1, top-2, top-3
    accuracies = evaluate_top_k_accuracy(chunk_size, overlap, k1, b, max_k=3)
    
    print("\n📈 RESULTS:")
    print("-" * 30)
    for k, accuracy in accuracies.items():
        print(f"Top-{k}: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Compare with original top-1 result
    original_accuracy = 0.945
    print(f"\n📊 Original top-1 result: {original_accuracy:.3f} ({original_accuracy*100:.1f}%)")
    
    if abs(accuracies[1] - original_accuracy) < 0.001:
        print("✅ Top-1 results match!")
    else:
        print(f"⚠️  Top-1 results differ by {abs(accuracies[1] - original_accuracy):.3f}")

if __name__ == "__main__":
    main()
