#!/usr/bin/env python3
"""
Search module for separated-models-2
BM25-only search with optimized chunking (chunk_size=128, overlap=12)
Based on Elias's improved results
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Configuration - optimized based on Elias's results
CHUNK_SIZE = 128
OVERLAP = 12
CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# Global cache
_BM25_CACHE = None

def chunk_words(words: List[str], size: int, overlap: int) -> List[str]:
    """Yield word windows of length *size* with given *overlap*."""
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i : i + size]
        if len(chunk) < 10:  # skip very small fragments
            continue
        chunks.append(" ".join(chunk))
        if i + size >= len(words):
            break
    return chunks

def build_bm25_index() -> Dict:
    """Build BM25 index with optimized parameters."""
    cache_path = CACHE_ROOT / f"bm25_index_condensed_{CHUNK_SIZE}_{OVERLAP}.pkl"
    
    if cache_path.exists():
        print(f"Loading cached BM25 index from {cache_path}")
        return pickle.loads(cache_path.read_bytes())

    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []
    topic_names: List[str] = []

    # Load topics mapping
    topics_data = load_topics_mapping()
    
    print(f"[bm25] Building index — condensed_topics size={CHUNK_SIZE} overlap={OVERLAP}")
    
    topic_dir = Path("data/condensed_topics")
    for md_file in topic_dir.rglob("*.md"):
        topic_name = md_file.parent.name
        topic_id = topics_data.get(topic_name, -1)
        
        if topic_id == -1:
            continue
            
        words = md_file.read_text(encoding="utf-8").split()
        for chunk_text in chunk_words(words, CHUNK_SIZE, OVERLAP):
            chunks.append(chunk_text)
            topics.append(topic_id)
            chunk_texts.append(chunk_text)
            topic_names.append(topic_name)

    # Build BM25 index
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    # Save everything
    data = {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'topic_names': topic_names,
        'bm25': bm25,
        'chunk_size': CHUNK_SIZE,
        'overlap': OVERLAP,
        'topics_data': topics_data
    }
    
    cache_path.write_bytes(pickle.dumps(data))
    print(f"[bm25] Cached index → {cache_path}")
    return data

def load_bm25_index() -> Dict:
    """Load or build BM25 index with caching."""
    global _BM25_CACHE
    
    if _BM25_CACHE is None:
        _BM25_CACHE = build_bm25_index()
    
    return _BM25_CACHE

def load_topics_mapping() -> Dict[str, int]:
    """Load topic name to ID mapping."""
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def bm25_search(statement: str, top_k: int = 1) -> List[Dict]:
    """Perform BM25 search and return top-k results."""
    data = load_bm25_index()
    tokenized_query = statement.lower().split()
    scores = data['bm25'].get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'topic_id': data['topics'][idx],
            'topic_name': data['topic_names'][idx],
            'chunk_text': data['chunk_texts'][idx],
            'score': scores[idx]
        })
    
    return results

def get_best_topic(statement: str) -> int:
    """
    Determine the best topic using BM25 search
    Returns the topic ID of the highest-scoring chunk
    """
    search_results = bm25_search(statement, top_k=1)
    
    if not search_results:
        return 0  # Default fallback
    
    # Return the topic of the highest-scoring result
    return search_results[0]['topic_id']

def get_rich_context_for_statement(statement: str, topic_id: int, max_chars: int = 4000) -> str:
    """
    Get rich context using BM25 search for the most relevant chunks
    """
    # Find topic name from ID
    topic_name = None
    topics_data = load_topics_mapping()
    for name, tid in topics_data.items():
        if tid == topic_id:
            topic_name = name
            break
    
    if not topic_name:
        return ""
    
    # Get the most relevant chunks for this statement
    search_results = bm25_search(statement, top_k=5)
    
    # Filter results to only include chunks from the identified topic
    relevant_chunks = []
    for result in search_results:
        if result['topic_id'] == topic_id:
            relevant_chunks.append(result['chunk_text'])
    
    # If no chunks from the topic, fall back to a few chunks from the topic
    if not relevant_chunks:
        # Get a few chunks from the topic as fallback
        all_results = bm25_search(statement, top_k=20)
        for result in all_results:
            if result['topic_id'] == topic_id:
                relevant_chunks.append(result['chunk_text'])
                if len(relevant_chunks) >= 3:  # Limit fallback chunks
                    break
    
    # Build context with only the most relevant chunks
    if relevant_chunks:
        context = f"MEDICAL CONTEXT FOR TOPIC: {topic_name}\n\n"
        context += "\n\n--- RELEVANT CHUNKS ---\n\n"
        context += "\n\n".join(relevant_chunks)
    else:
        # Fallback to topic name only if no chunks found
        context = f"MEDICAL CONTEXT FOR TOPIC: {topic_name}\n\n"
        context += "No specific relevant chunks found for this statement."
    
    # Truncate if too long
    if len(context) > max_chars:
        context = context[:max_chars] + "..."
    
    return context.strip()

def get_full_document_for_topic(topic_name: str) -> str:
    """
    Get the full document content for a specific topic
    """
    topics_dir = Path("data/condensed_topics")
    topic_dir = topics_dir / topic_name
    
    if not topic_dir.exists():
        return ""
    
    # Combine all markdown files for this topic
    full_content = ""
    for md_file in topic_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            full_content += f"\n\n--- {md_file.name} ---\n\n"
            full_content += content
    
    return full_content.strip()

def evaluate_bm25_performance():
    """Evaluate BM25 performance on test statements."""
    from pathlib import Path
    import json
    
    # Load test statements
    statements = []
    statement_dir = Path("data/train/statements")
    answer_dir = Path("data/train/answers")
    
    for path in sorted(statement_dir.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((answer_dir / f"statement_{sid}.json").read_text())
        statements.append((statement, ans["statement_topic"]))
    
    print(f"Testing BM25 search on {len(statements)} statements")
    print(f"Configuration: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")
    print("-" * 50)
    
    # Test top-k performance
    top_k_values = range(1, 11)
    
    for top_k in top_k_values:
        correct = 0
        for stmt, true_topic in tqdm(statements, desc=f"Top-{top_k}", leave=False):
            results = bm25_search(stmt, top_k=top_k)
            # Check if true topic is in any of the top-k results
            if any(result['topic_id'] == true_topic for result in results):
                correct += 1
        
        accuracy = correct / len(statements)
        print(f"Top-{top_k:2d}: {accuracy:.3f} ({correct}/{len(statements)})")
    
    print("-" * 50)

if __name__ == "__main__":
    # Test the BM25 search performance
    evaluate_bm25_performance() 