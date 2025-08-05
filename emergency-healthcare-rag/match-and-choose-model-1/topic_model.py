#!/usr/bin/env python3
"""
Search module for match-and-choose-model-1
BM25-only search with optimized chunking and enhanced scoring
Based on separated-models-2 with improvements for match-and-choose logic
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# Configuration - updated hyperparameters
CHUNK_SIZE = 96
OVERLAP = 12
BM25_K1 = 2.0
BM25_B = 1.2
USE_CONDENSED_TOPICS = False
FORCE_REBUILD = False  # Force rebuild BM25 index (ignore cache)
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
    topic_type = "condensed" if USE_CONDENSED_TOPICS else "regular"
    cache_path = CACHE_ROOT / f"bm25_index_{topic_type}_{CHUNK_SIZE}_{OVERLAP}.pkl"
    
    if cache_path.exists() and not FORCE_REBUILD:
        print(f"Loading cached BM25 index from {cache_path}")
        return pickle.loads(cache_path.read_bytes())
    
    if FORCE_REBUILD:
        print(f"🔄 Force rebuilding BM25 index (ignoring cache)")

    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []
    topic_names: List[str] = []

    # Load topics mapping
    topics_data = load_topics_mapping()
    
    topic_type = "condensed_topics" if USE_CONDENSED_TOPICS else "topics"
    print(f"[bm25] Building index — {topic_type} size={CHUNK_SIZE} overlap={OVERLAP}")
    
    topic_dir = Path(f"data/{topic_type}")
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

    # Build BM25 index with custom parameters
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks, k1=BM25_K1, b=BM25_B)

    # Save everything
    data = {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'topic_names': topic_names,
        'bm25': bm25,
        'chunk_size': CHUNK_SIZE,
        'overlap': OVERLAP,
        'use_condensed_topics': USE_CONDENSED_TOPICS,
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

def bm25_search(statement: str, top_k: int = 10) -> List[Dict]:
    """Perform BM25 search and return top-k results with scores."""
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
            'score': float(scores[idx])  # Convert to float for JSON serialization
        })
    
    return results

def get_top_topics_with_scores(statement: str, top_k: int = 10) -> List[Dict]:
    """
    Get top-k topics with their scores, deduplicated by topic_id
    Returns list of dicts with topic_id, topic_name, best_score, and best_chunk
    """
    search_results = bm25_search(statement, top_k=top_k*3)  # Get more results to account for duplicates
    
    # Group by topic_id and keep the best score for each topic
    topic_scores = {}
    for result in search_results:
        topic_id = result['topic_id']
        score = result['score']
        
        if topic_id not in topic_scores or score > topic_scores[topic_id]['score']:
            topic_scores[topic_id] = {
                'topic_id': topic_id,
                'topic_name': result['topic_name'],
                'score': score,
                'best_chunk': result['chunk_text']
            }
    
    # Sort by score and return top-k
    sorted_topics = sorted(topic_scores.values(), key=lambda x: x['score'], reverse=True)
    return sorted_topics[:top_k]

def get_best_topic(statement: str) -> int:
    """
    Determine the best topic using BM25 search
    Returns the topic ID of the highest-scoring chunk
    """
    top_topics = get_top_topics_with_scores(statement, top_k=1)
    
    if not top_topics:
        return 0  # Default fallback
    
    # Return the topic of the highest-scoring result
    return top_topics[0]['topic_id']

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
    topic_type = "condensed_topics" if USE_CONDENSED_TOPICS else "topics"
    topics_dir = Path(f"data/{topic_type}")
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