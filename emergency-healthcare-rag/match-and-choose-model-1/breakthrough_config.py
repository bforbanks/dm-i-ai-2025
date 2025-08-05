#!/usr/bin/env python3
"""
Implement the breakthrough configuration that achieved 91% accuracy
Found: all-distilroberta-v1 + linear_0.7 (chunk_size=96, overlap=9) → 91.0%
"""

import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

# Import required libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("📦 Install: pip install sentence-transformers scikit-learn rank_bm25")
    exit(1)

# -----------------------------------------------------------------------
# PATHS AND CONSTANTS
# -----------------------------------------------------------------------

CONDENSED_TOPIC_DIR = Path("data/condensed_topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())
CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# BREAKTHROUGH CONFIGURATION
BREAKTHROUGH_CONFIG = {
    'model': 'sentence-transformers/all-distilroberta-v1',
    'chunk_size': 96,
    'overlap': 9,
    'strategy': 'linear_0.7',  # 70% semantic, 30% BM25
    'use_condensed_topics': True
}

print(f"🚀 IMPLEMENTING BREAKTHROUGH CONFIGURATION")
print(f"=" * 50)
print(f"🧠 Model: {BREAKTHROUGH_CONFIG['model']}")
print(f"📊 Strategy: {BREAKTHROUGH_CONFIG['strategy']} (70% semantic + 30% BM25)")
print(f"⚙️  BM25: chunk_size={BREAKTHROUGH_CONFIG['chunk_size']}, overlap={BREAKTHROUGH_CONFIG['overlap']}")
print(f"📁 Topics: {'Condensed' if BREAKTHROUGH_CONFIG['use_condensed_topics'] else 'Original'}")

# -----------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------

def chunk_words(words: List[str], size: int, overlap: int):
    """Create overlapping word chunks"""
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        yield words[i : i + size]
        if i + size >= len(words):
            break

def build_bm25_index(chunk_size: int, overlap: int) -> Dict:
    """Build BM25 index for breakthrough configuration"""
    cache_path = CACHE_ROOT / f"bm25_index_condensed_topics_{chunk_size}_{overlap}.pkl"
    
    if cache_path.exists():
        print(f"📚 Loading cached BM25 index...")
        try:
            data = pickle.loads(cache_path.read_bytes())
            print(f"✅ Loaded BM25 index with {len(data['chunks'])} chunks")
            return data
        except:
            print(f"⚠️  Cache corrupted, rebuilding...")
            cache_path.unlink()
    
    print(f"🔨 Building BM25 index (chunk_size={chunk_size}, overlap={overlap})...")
    chunks = []
    topics = []
    
    for md_file in CONDENSED_TOPIC_DIR.rglob("*.md"):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP[topic_name]
        words = md_file.read_text(encoding="utf-8").split()
        
        for w_chunk in chunk_words(words, chunk_size, overlap):
            if len(w_chunk) < 10:  # Skip tiny fragments
                continue
            chunk_text = " ".join(w_chunk)
            chunks.append(chunk_text)
            topics.append(topic_id)
    
    # Build BM25 index
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    data = {
        'chunks': chunks,
        'topics': topics,
        'bm25': bm25,
        'chunk_size': chunk_size,
        'overlap': overlap
    }
    
    # Cache the results
    cache_path.write_bytes(pickle.dumps(data))
    print(f"✅ Built and cached BM25 index with {len(chunks)} chunks")
    return data

def build_semantic_index(model_name: str, bm25_data: Dict) -> Dict:
    """Build semantic embeddings for breakthrough configuration"""
    chunks = bm25_data['chunks']
    cache_path = CACHE_ROOT / f"semantic_index_{model_name.replace('/', '_')}_condensed_{bm25_data['chunk_size']}_{bm25_data['overlap']}.pkl"
    
    if cache_path.exists():
        print(f"📚 Loading cached semantic index...")
        try:
            data = pickle.loads(cache_path.read_bytes())
            print(f"✅ Loaded semantic embeddings for {len(data['embeddings'])} chunks")
            return data
        except:
            print(f"⚠️  Cache corrupted, rebuilding...")
            cache_path.unlink()
    
    print(f"🧠 Building semantic index with {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
    
    embeddings = model.encode(chunks, show_progress_bar=True, batch_size=32)
    
    data = {
        'embeddings': embeddings,
        'model_name': model_name,
        'chunk_size': bm25_data['chunk_size'],
        'overlap': bm25_data['overlap']
    }
    
    cache_path.write_bytes(pickle.dumps(data))
    print(f"✅ Built and cached semantic index")
    return data

def load_statements() -> List[Tuple[str, int]]:
    """Load evaluation statements"""
    statements = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        answer = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        statements.append((statement, answer["statement_topic"]))
    return statements

def hybrid_search(query: str, bm25_data: Dict, semantic_data: Dict, strategy: str, top_k: int = 5) -> List[Dict]:
    """Perform hybrid search using breakthrough strategy"""
    chunks = bm25_data['chunks']
    topics = bm25_data['topics']
    
    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_data['bm25'].get_scores(tokenized_query)
    
    # Semantic search
    model = SentenceTransformer(semantic_data['model_name'])
    query_embedding = model.encode([query])
    semantic_scores = cosine_similarity(query_embedding, semantic_data['embeddings'])[0]
    
    # Linear fusion (strategy = linear_0.7 means 70% semantic, 30% BM25)
    if strategy.startswith('linear_'):
        semantic_weight = float(strategy.split('_')[1])
        bm25_weight = 1.0 - semantic_weight
        
        # Normalize scores to [0,1]
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        # Combine scores
        final_scores = semantic_weight * semantic_scores_norm + bm25_weight * bm25_scores_norm
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Get top-k results
    top_indices = np.argsort(final_scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'topic_id': topics[idx],
            'chunk_text': chunks[idx],
            'final_score': final_scores[idx],
            'bm25_score': bm25_scores[idx],
            'semantic_score': semantic_scores[idx]
        })
    
    return results

def evaluate_breakthrough_config():
    """Evaluate the breakthrough configuration"""
    print(f"\n🔬 EVALUATING BREAKTHROUGH CONFIGURATION")
    print(f"=" * 50)
    
    # Build indices
    bm25_data = build_bm25_index(
        BREAKTHROUGH_CONFIG['chunk_size'], 
        BREAKTHROUGH_CONFIG['overlap']
    )
    
    semantic_data = build_semantic_index(
        BREAKTHROUGH_CONFIG['model'], 
        bm25_data
    )
    
    # Load evaluation data
    statements = load_statements()
    print(f"📊 Evaluating on {len(statements)} statements...")
    
    # Evaluate
    results = []
    correct_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    reciprocal_ranks = []
    
    start_time = time.time()
    for stmt, true_topic in tqdm(statements, desc="Evaluating"):
        search_results = hybrid_search(
            stmt, bm25_data, semantic_data, 
            BREAKTHROUGH_CONFIG['strategy'], top_k=5
        )
        
        # Calculate metrics
        predicted_topics = [r['topic_id'] for r in search_results]
        
        # Top-k accuracy
        for k in range(1, 6):
            if true_topic in predicted_topics[:k]:
                correct_counts[k] += 1
        
        # Mean Reciprocal Rank
        try:
            rank = predicted_topics.index(true_topic) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
        
        results.append({
            'statement': stmt,
            'true_topic': true_topic,
            'predicted_topics': predicted_topics,
            'scores': [r['final_score'] for r in search_results]
        })
    
    elapsed = time.time() - start_time
    
    # Calculate final metrics
    total = len(statements)
    metrics = {
        'top1_accuracy': correct_counts[1] / total,
        'top2_accuracy': correct_counts[2] / total, 
        'top3_accuracy': correct_counts[3] / total,
        'top4_accuracy': correct_counts[4] / total,
        'top5_accuracy': correct_counts[5] / total,
        'mrr': np.mean(reciprocal_ranks),
        'evaluation_time': elapsed,
        'time_per_query': elapsed / total
    }
    
    # Display results
    print(f"\n🎉 BREAKTHROUGH CONFIGURATION RESULTS")
    print(f"=" * 50)
    print(f"🧠 Model: {BREAKTHROUGH_CONFIG['model'].split('/')[-1]}")
    print(f"🔍 Strategy: {BREAKTHROUGH_CONFIG['strategy']}")
    print(f"⚙️  BM25: chunk_size={BREAKTHROUGH_CONFIG['chunk_size']}, overlap={BREAKTHROUGH_CONFIG['overlap']}")
    print(f"\n📊 ACCURACY METRICS:")
    print(f"   Top-1: {metrics['top1_accuracy']:.3f} ({correct_counts[1]}/{total})")
    print(f"   Top-2: {metrics['top2_accuracy']:.3f} ({correct_counts[2]}/{total})")
    print(f"   Top-3: {metrics['top3_accuracy']:.3f} ({correct_counts[3]}/{total})")
    print(f"   Top-4: {metrics['top4_accuracy']:.3f} ({correct_counts[4]}/{total})")
    print(f"   Top-5: {metrics['top5_accuracy']:.3f} ({correct_counts[5]}/{total})")
    print(f"   MRR: {metrics['mrr']:.3f}")
    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   Time per query: {metrics['time_per_query']:.3f}s")
    
    # Save results
    output_file = "breakthrough_config_results.json"
    output_data = {
        'config': BREAKTHROUGH_CONFIG,
        'metrics': metrics,
        'detailed_results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return metrics

if __name__ == "__main__":
    evaluate_breakthrough_config()