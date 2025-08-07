#!/usr/bin/env python3
"""
BM25 + LLM Reranking Pipeline
Step 1: BM25 on original topics to find top-3 chunks
Step 2: Deduplicate parent topics
Step 3: LLM reranking on summarized topics
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

def build_bm25_index(chunk_size: int, overlap: int) -> Dict:
    """Build BM25 index on original topics."""
    cache_path = CACHE_ROOT / f"bm25_original_{chunk_size}_{overlap}.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []

    print(f"[bm25] Building index on original topics — size={chunk_size} overlap={overlap}")
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

    # Build BM25 index
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
    bm25 = BM25Okapi(tokenized_chunks)

    data = {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'bm25': bm25,
        'chunk_size': chunk_size,
        'overlap': overlap
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

def llm_rerank(statement: str, candidate_topics: List[Tuple[str, str]]) -> str:
    """Use LLM to rerank topics based on summarized content."""
    try:
        import requests
        import json
        
        prompt = create_rerank_prompt(statement, candidate_topics)
        print(f"Calling Ollama API with model: gemma3:27b")
        print(f"Prompt length: {len(prompt)} characters")
        
        # Call Ollama API
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gemma3:27b",  # Using larger model for better performance
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent responses
                "top_p": 0.9,
                "num_predict": 50  # Limit response length
            }
        }
        
        print("Sending request to Ollama...")
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        llm_response = result.get('response', '').strip()
        print(f"Ollama response: '{llm_response}'")
        
        # Parse the response to extract topic name
        # The LLM should return just the topic name
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
        print(f"Could not parse LLM response: '{llm_response}'")
        return candidate_topics[0][0] if candidate_topics else "Unknown"
        
    except Exception as e:
        print(f"LLM reranking failed: {e}")
        # Fallback to first topic
        return candidate_topics[0][0] if candidate_topics else "Unknown"

def bm25_with_llm_rerank(statement: str) -> int:
    """
    BM25 + LLM Reranking Pipeline:
    1. BM25 returns top-3 chunks from original topics
    2. Deduplicate topic_ids
    3. Load summarized topics for LLM reranking
    4. LLM chooses best topic
    """
    # Step 1: BM25 returns top-3 chunks from original topics
    data = build_bm25_index(128, 12)  # Fixed optimal config
    bm25_results = bm25_search(statement, data, top_k=3)
    
    # Step 2: Deduplicate topic_ids
    unique_topics = {}
    for result in bm25_results:
        topic_id = result['topic_id']
        if topic_id not in unique_topics:
            unique_topics[topic_id] = result['score']
        else:
            # Keep the higher score
            unique_topics[topic_id] = max(unique_topics[topic_id], result['score'])
    
    # Debug: Always show what BM25 found
    print(f"BM25 found {len(unique_topics)} unique topics:")
    for topic_id, score in unique_topics.items():
        topic_name = get_topic_name_from_id(topic_id)
        print(f"  - {topic_name} (ID: {topic_id}, Score: {score:.3f})")
    
    # If only one unique topic, still call LLM for debugging
    if len(unique_topics) == 1:
        print("Only one topic found, but calling LLM anyway for testing...")
        topic_id = list(unique_topics.keys())[0]
        topic_name = get_topic_name_from_id(topic_id)
        summary = load_summarized_topic(topic_name)
        candidate_topics = [(topic_name, summary)]
        
        # Call LLM anyway
        best_topic_name = llm_rerank(statement, candidate_topics)
        print(f"LLM selected: {best_topic_name}")
        return topic_id
    
    # Step 3: Load summarized topics for LLM reranking
    candidate_topics = []
    for topic_id, score in unique_topics.items():
        topic_name = get_topic_name_from_id(topic_id)
        summary = load_summarized_topic(topic_name)
        candidate_topics.append((topic_name, summary))
    
    # Debug: Print candidate topics
    print(f"BM25 found {len(unique_topics)} unique topics:")
    for topic_name, summary in candidate_topics:
        print(f"  - {topic_name}")
    
    # Step 4: LLM chooses best topic
    best_topic_name = llm_rerank(statement, candidate_topics)
    print(f"LLM selected: {best_topic_name}")
    
    # Convert back to topic_id
    for topic_name, tid in TOPIC_MAP.items():
        if topic_name == best_topic_name:
            return tid
    
    # Fallback to first topic if conversion fails
    return list(unique_topics.keys())[0] if unique_topics else 0

def load_statements():
    """Load test statements."""
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records

def evaluate_pipeline():
    """Evaluate the BM25 + LLM reranking pipeline."""
    statements = load_statements()
    
    print(f"=== BM25 + LLM RERANKING EVALUATION ===")
    print(f"BM25: Original topics (chunk_size=128, overlap=12)")
    print(f"LLM: Summarized topics for reranking")
    print("-" * 60)
    
    correct = 0
    total = len(statements)
    
    for stmt, true_topic in tqdm(statements, desc="Evaluating"):
        predicted_topic = bm25_with_llm_rerank(stmt)
        if predicted_topic == true_topic:
            correct += 1
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="BM25 + LLM Reranking Pipeline")
    parser.add_argument(
        "--evaluate", 
        action="store_true",
        help="Evaluate the pipeline on test statements"
    )
    parser.add_argument(
        "--test-statement", 
        type=str,
        help="Test a single statement"
    )
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_pipeline()
    elif args.test_statement:
        predicted_topic = bm25_with_llm_rerank(args.test_statement)
        topic_name = get_topic_name_from_id(predicted_topic)
        print(f"Statement: {args.test_statement}")
        print(f"Predicted topic: {topic_name} (ID: {predicted_topic})")
    else:
        print("Use --evaluate to test on all statements or --test-statement 'your statement' to test a single statement")

if __name__ == "__main__":
    main() 