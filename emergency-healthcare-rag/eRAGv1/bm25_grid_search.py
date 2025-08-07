#!/usr/bin/env python3
"""
BM25 Grid Search with k1 and b parameters - Top-1 Focus
Uses original topic MD files with comprehensive parameter optimization
"""

import argparse
from pathlib import Path
import json, pickle
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import itertools

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIGINAL_TOPIC_DIR = Path("data/topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())

CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# Grid search parameters
# CHUNK_SIZES = [64, 128, 256, 512]  # Must include 128
# CHUNK_OVERLAPS = [8, 12, 24, 48]   # Must include 12
CHUNK_SIZES = [128, 96]                    # 96 for tighter chunks
CHUNK_OVERLAPS = [12, 48]                  # high overlap for tighter context
K1_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]      # push it further — 2.5 can help if TF scaling isn't flat
B_VALUES  = [0.5,0.75, 0.95, 1.0, 1.2]  # b > 1.0? yes, let’s break the rules

# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def chunk_words(words: List[str], size: int, overlap: int):
    """Yield word windows of length *size* with given *overlap*."""
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        yield words[i : i + size]
        if i + size >= len(words):
            break

def build_bm25_index(chunk_size: int, overlap: int, k1: float, b: float) -> Dict:
    """Build BM25 index on original topics with specified parameters."""
    cache_path = CACHE_ROOT / f"bm25_original_{chunk_size}_{overlap}_{k1}_{b}.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []

    print(f"[bm25] Building index — size={chunk_size} overlap={overlap} k1={k1} b={b}")
    for md_file in ORIGINAL_TOPIC_DIR.rglob("*.md"):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP[topic_name]
        words = md_file.read_text(encoding="utf-8").split()
        for w_chunk in chunk_words(words, chunk_size, overlap):
            if len(w_chunk) < 10:  # skip very small fragments
                continue
            chunk_text = " ".join(w_chunk)
            chunks.append(chunk_text)
            topics.append(topic_id)
            chunk_texts.append(chunk_text)

    # Build BM25 index with custom k1 and b
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks, k1=k1, b=b)

    data = {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'bm25': bm25,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'k1': k1,
        'b': b
    }
    
    cache_path.write_bytes(pickle.dumps(data))
    print(f"[bm25] Cached index → {cache_path}")
    return data

def bm25_search(statement: str, data: Dict, top_k: int = 1) -> List[Dict]:
    """Perform BM25 search and return top-k results."""
    tokenized_query = statement.lower().split()
    scores = data['bm25'].get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'topic_id': data['topics'][idx],
            'chunk_text': data['chunk_texts'][idx],
            'score': scores[idx]
        })
    
    return results

def load_statements():
    """Load test statements."""
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records

def evaluate_config(chunk_size: int, overlap: int, k1: float, b: float) -> float:
    """Evaluate a specific BM25 configuration for top-1 accuracy."""
    statements = load_statements()
    
    # Build index
    data = build_bm25_index(chunk_size, overlap, k1, b)
    
    # Evaluate top-1 accuracy
    correct = 0
    total = len(statements)
    
    for stmt, true_topic in statements:
        results = bm25_search(stmt, data, top_k=1)
        if results and results[0]['topic_id'] == true_topic:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def grid_search():
    """Perform comprehensive grid search over all parameters for top-1 accuracy."""
    print("=== BM25 GRID SEARCH (TOP-1 ACCURACY) ===")
    print(f"Chunk sizes: {CHUNK_SIZES}")
    print(f"Chunk overlaps: {CHUNK_OVERLAPS}")
    print(f"k1 values: {K1_VALUES}")
    print(f"b values: {B_VALUES}")
    print(f"Total combinations: {len(CHUNK_SIZES) * len(CHUNK_OVERLAPS) * len(K1_VALUES) * len(B_VALUES)}")
    print("-" * 60)
    
    # Generate all parameter combinations (reversed order: b, k1, overlap, chunk_size)
    param_combinations = list(itertools.product(B_VALUES, K1_VALUES, CHUNK_OVERLAPS, CHUNK_SIZES))
    
    results = []
    
    for b, k1, overlap, chunk_size in tqdm(param_combinations, desc="Grid search", ncols=80):
        accuracy = evaluate_config(chunk_size, overlap, k1, b)
        results.append({
            'chunk_size': chunk_size,
            'overlap': overlap,
            'k1': k1,
            'b': b,
            'accuracy': accuracy
        })
        print(f"size={chunk_size:3d} overlap={overlap:2d} k1={k1:3.1f} b={b:3.1f} → {accuracy:.3f}")
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("\n=== TOP 10 RESULTS (TOP-1 ACCURACY) ===")
    for i, result in enumerate(results[:10]):
        print(f"{i+1:2d}. size={result['chunk_size']:3d} overlap={result['overlap']:2d} "
              f"k1={result['k1']:3.1f} b={result['b']:3.1f} → {result['accuracy']:.3f}")
    
    # Save results
    results_file = CACHE_ROOT / "grid_search_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="BM25 Grid Search with k1 and b parameters - Top-1 Focus")
    parser.add_argument(
        "--grid-search", 
        action="store_true",
        help="Perform comprehensive grid search over all parameters"
    )
    
    args = parser.parse_args()
    
    if args.grid_search:
        grid_search()
    else:
        print("Use --grid-search to perform comprehensive parameter optimization")
        print("This will test all combinations and find the best top-1 accuracy")

if __name__ == "__main__":
    main() 