#!/usr/bin/env python3
"""
Fine-tune BM25 search with smaller chunk sizes and different overlap ratios
"""

from pathlib import Path
import json, pickle
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
CONDENSED_TOPIC_DIR = Path("data/condensed_topics")
ORIGINAL_TOPIC_DIR = Path("data/topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())

CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunk_words(words: List[str], size: int, overlap: int):
    """Yield word windows of length *size* with given *overlap*."""
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        yield words[i : i + size]
        if i + size >= len(words):
            break


def build_bm25_index(chunk_size: int, overlap: int, topic_dir: Path) -> Dict:
    """Build BM25 index for given chunk parameters."""
    dir_name = topic_dir.name
    cache_path = CACHE_ROOT / f"bm25_index_{dir_name}_{chunk_size}_{overlap}.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []

    print(f"[bm25] Building index — {dir_name} size={chunk_size} overlap={overlap}")
    for md_file in topic_dir.rglob("*.md"):
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

    # Build BM25 index
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    # Save everything
    data = {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'bm25': bm25,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'topic_dir': dir_name
    }
    
    cache_path.write_bytes(pickle.dumps(data))
    print(f"[bm25] Cached index → {cache_path}")
    return data


def load_statements():
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records


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


def evaluate_bm25_fine_tune():
    """Evaluate BM25 with various chunk sizes and overlap ratios."""
    statements = load_statements()
    
    # Test different chunk sizes and overlap ratios
    chunk_sizes = [32, 48, 64, 80, 96, 112, 128]
    overlap_ratios = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]  # as fraction of chunk size
    
    print("=== BM25 FINE-TUNING RESULTS ===")
    print("Format: chunk_size (overlap) → accuracy")
    print("-" * 50)
    
    best_config = None
    best_accuracy = 0
    
    for cs in chunk_sizes:
        for ratio in overlap_ratios:
            ov = int(cs * ratio)
            if ov >= cs:  # skip invalid overlaps
                continue
                
            try:
                data = build_bm25_index(cs, ov)
                
                correct = 0
                for stmt, true_topic in tqdm(statements, desc=f"cs={cs} ov={ov}", leave=False):
                    results = bm25_search(stmt, data, top_k=1)
                    if results and results[0]['topic_id'] == true_topic:
                        correct += 1
                
                acc = correct / len(statements)
                print(f"chunk={cs:3d} overlap={ov:3d} ({ratio:.1f}) → {acc:.3f} ({correct}/{len(statements)})")
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_config = (cs, ov, ratio)
                    
            except Exception as e:
                print(f"Error with cs={cs} ov={ov}: {e}")
                continue
    
    print("-" * 50)
    print(f"BEST CONFIG: chunk={best_config[0]} overlap={best_config[1]} (ratio={best_config[2]:.1f})")
    print(f"BEST ACCURACY: {best_accuracy:.3f}")
    
    # Test top-k performance for best configuration
    print(f"\n=== TOP-K PERFORMANCE FOR BEST CONFIG ===")
    test_top_k_performance(best_config[0], best_config[1])


def test_best_config_both_topics():
    """Test the best config on both condensed and original topics."""
    # Best config from previous results
    chunk_size = 128
    overlap = 12
    
    print("=== COMPARING CONDENSED vs ORIGINAL TOPICS ===")
    print(f"Using best config: chunk={chunk_size}, overlap={overlap}")
    print("=" * 60)
    
    # # Test condensed topics
    # print("\n--- CONDENSED TOPICS ---")
    # condensed_data = build_bm25_index(chunk_size, overlap, CONDENSED_TOPIC_DIR)
    # test_top_k_performance_with_data(condensed_data, "condensed_topics")
    
    # Test original topics
    print("\n--- ORIGINAL TOPICS ---")
    original_data = build_bm25_index(chunk_size, overlap, ORIGINAL_TOPIC_DIR)
    test_top_k_performance_with_data(original_data, "original_topics")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)


def test_top_k_performance_with_data(data: Dict, topic_name: str):
    """Test top-k performance for a specific configuration."""
    statements = load_statements()
    
    top_k_values = range(1,11)
    
    print(f"Testing {topic_name}")
    print("-" * 40)
    
    results_dict = {}
    for top_k in top_k_values:
        correct = 0
        for stmt, true_topic in tqdm(statements, desc=f"Top-{top_k}", leave=False):
            results = bm25_search(stmt, data, top_k=top_k)
            # Check if true topic is in any of the top-k results
            if any(result['topic_id'] == true_topic for result in results):
                correct += 1
        
        accuracy = correct / len(statements)
        results_dict[top_k] = accuracy
        print(f"Top-{top_k:2d}: {accuracy:.3f} ({correct}/{len(statements)})")
    
    print("-" * 40)
    return results_dict


if __name__ == "__main__":
    # evaluate_bm25_fine_tune()  # Comment out the full evaluation
    test_best_config_both_topics()   # Compare condensed vs original topics 