#!/usr/bin/env python3
"""
Simple BM25 Search on Original Topics with LLM Reranking
Uses optimal configuration: chunk_size=128, overlap=12
Implements 5% threshold: if 2nd/3rd place are within 5% of top score, use LLM
"""

import argparse
from pathlib import Path
import json, pickle
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIGINAL_TOPIC_DIR = Path("data/topics")
SUMMARIZED_TOPIC_DIR = Path("data/summarized_topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())

CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# Optimal configuration (already found)
CHUNK_SIZE = 128
OVERLAP = 12

# Threshold for LLM reranking (5%)
THRESHOLD_PERCENT = 5.0

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

def build_bm25_index() -> Dict:
    """Build BM25 index on original topics with optimal config."""
    cache_path = CACHE_ROOT / f"bm25_original_{CHUNK_SIZE}_{OVERLAP}.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []

    print(f"[bm25] Building index on original topics — size={CHUNK_SIZE} overlap={OVERLAP}")
    for md_file in ORIGINAL_TOPIC_DIR.rglob("*.md"):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP[topic_name]
        words = md_file.read_text(encoding="utf-8").split()
        for w_chunk in chunk_words(words, CHUNK_SIZE, OVERLAP):
            if len(w_chunk) < 10:  # skip very small fragments
                continue
            chunk_text = " ".join(w_chunk)
            chunks.append(chunk_text)
            topics.append(topic_id)
            chunk_texts.append(chunk_text)

    # Build BM25 index
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    data = {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'bm25': bm25,
        'chunk_size': CHUNK_SIZE,
        'overlap': OVERLAP
    }
    
    cache_path.write_bytes(pickle.dumps(data))
    print(f"[bm25] Cached index → {cache_path}")
    return data

def bm25_search(statement: str, data: Dict, top_k: int = 3) -> List[Dict]:
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

def get_topic_name_from_id(topic_id: int) -> str:
    """Get topic name from topic ID."""
    for name, tid in TOPIC_MAP.items():
        if tid == topic_id:
            return name
    return f"Unknown_{topic_id}"

def load_summarized_topic(topic_name: str) -> str:
    """Load summarized topic content."""
    topic_file = SUMMARIZED_TOPIC_DIR / topic_name / f"{topic_name}.md"
    if topic_file.exists():
        return topic_file.read_text(encoding="utf-8").strip()
    else:
        return f"Summarized topic not found for: {topic_name}"

def create_rerank_prompt(statement: str, candidate_topics: List[Tuple[str, str]]) -> str:
    """Create prompt for LLM reranking."""
    prompt = """You are a medical triage assistant. Given a medical statement, choose the most relevant topic summary.

### Statement:
"{statement}"

### Candidate Topics:

""".format(statement=statement)

    for i, (topic_name, summary) in enumerate(candidate_topics, 1):
        # Truncate summary if too long (keep first 500 chars)
        truncated_summary = summary[:500] + "..." if len(summary) > 500 else summary
        prompt += f"**{i}. {topic_name}**\n{truncated_summary}\n\n"

    prompt += """### Question:
Which topic best matches the statement? Return the topic name only.

Answer:"""
    
    return prompt

def llm_rerank(statement: str, candidate_topics: List[Tuple[str, str]], verbose: bool = True) -> str:
    """Use LLM to rerank topics based on summarized content."""
    try:
        import requests
        import json
        
        prompt = create_rerank_prompt(statement, candidate_topics)
        if verbose:
            print(f"Calling Ollama API with model: gemma3:27b")
            print(f"Prompt length: {len(prompt)} characters")
        
        # Call Ollama API
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gemma3:27b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 50
            }
        }
        
        if verbose:
            print("Sending request to Ollama...")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        llm_response = result.get('response', '').strip()
        if verbose:
            print(f"Ollama response: '{llm_response}'")
        
        # Parse the response to extract topic name
        llm_response = llm_response.replace('Answer:', '').strip()
        
        # Check if the response matches any of our candidate topics
        for topic_name, _ in candidate_topics:
            if topic_name.lower() in llm_response.lower():
                return topic_name
        
        # If no exact match, try to find the closest match
        for topic_name, _ in candidate_topics:
            if any(word in llm_response.lower() for word in topic_name.lower().split()):
                return topic_name
        
        # Fallback to first topic if parsing fails
        if verbose:
            print(f"Could not parse LLM response: '{llm_response}'")
        return candidate_topics[0][0] if candidate_topics else "Unknown"
        
    except Exception as e:
        if verbose:
            print(f"LLM reranking failed: {e}")
        # Fallback to first topic
        return candidate_topics[0][0] if candidate_topics else "Unknown"

def bm25_with_threshold_rerank(statement: str, verbose: bool = True) -> int:
    """
    BM25 + Threshold-based LLM Reranking:
    1. BM25 returns top-3 results
    2. Check if 2nd/3rd place are within 5% of top score
    3. If close, use LLM to choose between close candidates
    4. If not close, just use BM25's top result
    """
    data = build_bm25_index()
    results = bm25_search(statement, data, top_k=3)
    
    if not results:
        return 0
    
    # Get top score
    top_score = results[0]['score']
    top_topic_id = results[0]['topic_id']
    
    # Calculate threshold
    threshold = top_score * (THRESHOLD_PERCENT / 100.0)
    min_score_for_llm = top_score - threshold
    
    if verbose:
        print(f"Top score: {top_score:.3f}")
        print(f"Threshold ({THRESHOLD_PERCENT}%): {threshold:.3f}")
        print(f"Min score for LLM: {min_score_for_llm:.3f}")
    
    # Find close candidates (within threshold)
    close_candidates = []
    for result in results:
        if result['score'] >= min_score_for_llm:
            topic_name = get_topic_name_from_id(result['topic_id'])
            summary = load_summarized_topic(topic_name)
            close_candidates.append((topic_name, summary))
            if verbose:
                print(f"  Close candidate: {topic_name} (Score: {result['score']:.3f})")
    
    # If only one candidate or no close candidates, use BM25 top result
    if len(close_candidates) <= 1:
        if verbose:
            print(f"Using BM25 top result: {get_topic_name_from_id(top_topic_id)}")
        return top_topic_id
    
    # If multiple close candidates, use LLM to choose
    if verbose:
        print(f"Multiple close candidates ({len(close_candidates)}), using LLM to choose...")
    best_topic_name = llm_rerank(statement, close_candidates, verbose=verbose)
    if verbose:
        print(f"LLM selected: {best_topic_name}")
    
    # Convert back to topic_id
    for topic_name, tid in TOPIC_MAP.items():
        if topic_name == best_topic_name:
            return tid
    
    # Fallback to top BM25 result
    return top_topic_id

def load_statements():
    """Load test statements."""
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records

def evaluate_bm25():
    """Evaluate BM25 on original topics with optimal config."""
    statements = load_statements()
    
    print(f"=== BM25 EVALUATION (ORIGINAL TOPICS) ===")
    print(f"Config: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")
    print("-" * 50)
    
    # Build index
    data = build_bm25_index()
    
    # Evaluate
    correct = 0
    total = len(statements)
    
    for stmt, true_topic in tqdm(statements, desc="Evaluating"):
        results = bm25_search(stmt, data, top_k=1)
        if results and results[0]['topic_id'] == true_topic:
            correct += 1
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    return accuracy

def evaluate_threshold_rerank():
    """Evaluate BM25 + Threshold-based LLM Reranking."""
    statements = load_statements()
    
    print(f"=== BM25 + THRESHOLD LLM RERANKING EVALUATION ===")
    print(f"Config: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")
    print(f"Threshold: {THRESHOLD_PERCENT}%")
    print("-" * 60)
    
    # Evaluate
    correct = 0
    total = len(statements)
    llm_calls = 0
    
    for stmt, true_topic in tqdm(statements, desc="Evaluating"):
        predicted_topic = bm25_with_threshold_rerank(stmt, verbose=False)
        if predicted_topic == true_topic:
            correct += 1
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    return accuracy

def test_top_k_performance():
    """Test top-k performance."""
    statements = load_statements()
    data = build_bm25_index()
    
    print(f"=== TOP-K PERFORMANCE ===")
    print(f"Config: chunk_size={CHUNK_SIZE}, overlap={OVERLAP}")
    print("-" * 40)
    
    top_k_values = range(1, 11)
    
    for top_k in top_k_values:
        correct = 0
        for stmt, true_topic in tqdm(statements, desc=f"Top-{top_k}", leave=False):
            results = bm25_search(stmt, data, top_k=top_k)
            # Check if true topic is in any of the top-k results
            if any(result['topic_id'] == true_topic for result in results):
                correct += 1
        
        accuracy = correct / len(statements)
        print(f"Top-{top_k:2d}: {accuracy:.3f} ({correct}/{len(statements)})")

def main():
    parser = argparse.ArgumentParser(description="Simple BM25 Search on Original Topics with LLM Reranking")
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Evaluate BM25 on test statements"
    )
    parser.add_argument(
        "--evaluate-threshold", 
        action="store_true",
        help="Evaluate BM25 + Threshold-based LLM Reranking"
    )
    parser.add_argument(
        "--top-k", 
        action="store_true",
        help="Test top-k performance"
    )
    parser.add_argument(
        "--test-statement", 
        type=str,
        help="Test a single statement with threshold reranking"
    )
    parser.add_argument(
        "--top-k-value", 
        type=int,
        default=1,
        help="Number of top results to return (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Handle multiple flags
    if args.evaluate:
        evaluate_bm25()
        print()  # Add spacing
    
    if args.evaluate_threshold:
        evaluate_threshold_rerank()
        print()  # Add spacing
    
    if args.top_k:
        test_top_k_performance()
        print()  # Add spacing
    
    if args.test_statement:
        print(f"Statement: {args.test_statement}")
        print(f"Using {THRESHOLD_PERCENT}% threshold for LLM reranking")
        print("-" * 50)
        
        predicted_topic = bm25_with_threshold_rerank(args.test_statement)
        topic_name = get_topic_name_from_id(predicted_topic)
        print(f"\nFinal prediction: {topic_name} (ID: {predicted_topic})")
    
    # If no flags provided, show help
    if not any([args.evaluate, args.evaluate_threshold, args.top_k, args.test_statement]):
        print("Use --evaluate to test BM25 only")
        print("Use --evaluate-threshold to test BM25 + LLM reranking with 5% threshold")
        print("Use --top-k to test top-k performance")
        print("Use --test-statement 'your statement' to test a single statement")
        print("You can combine flags: --evaluate --evaluate-threshold --top-k")

if __name__ == "__main__":
    main() 