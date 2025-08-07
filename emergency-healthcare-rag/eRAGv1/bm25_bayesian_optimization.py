#!/usr/bin/env python3
"""
BM25 Bayesian Optimization using Optuna
Efficiently searches the 4-dimensional hyperparameter space
"""

import argparse
from pathlib import Path
import json, pickle
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import optuna
from optuna.samplers import TPESampler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ORIGINAL_TOPIC_DIR = Path("data/topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())

CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# Parameter bounds for BO
PARAM_BOUNDS = {
    'chunk_size': (64, 256),      # Reasonable chunk size range
    'overlap': (8, 64),           # Reasonable overlap range
    'k1': (0.1, 3.0),            # BM25 k1 parameter range
    'b': (0.1, 1.5)              # BM25 b parameter range (including >1)
}

# Exploration strategy
EXPLORATION_RATIO = 0.3  # First 30% of trials focus on exploration

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
    # Round parameters to reasonable values for caching
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    k1 = round(k1, 2)
    b = round(b, 2)
    
    # No caching - build fresh each time to avoid 1000+ files

    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []

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

def objective(trial):
    """Objective function for Optuna optimization."""
    # Determine if we're in exploration phase - use a simpler approach
    is_exploration = trial.number < 300  # First 300 trials for exploration (30% of 1000)
    
    # Suggest hyperparameters with different strategies
    if is_exploration:
        # High exploration: use uniform sampling
        chunk_size = trial.suggest_int('chunk_size', 
                                      PARAM_BOUNDS['chunk_size'][0], 
                                      PARAM_BOUNDS['chunk_size'][1])
        overlap = trial.suggest_int('overlap', 
                                   PARAM_BOUNDS['overlap'][0], 
                                   PARAM_BOUNDS['overlap'][1])
        k1 = trial.suggest_float('k1', 
                                PARAM_BOUNDS['k1'][0], 
                                PARAM_BOUNDS['k1'][1])
        b = trial.suggest_float('b', 
                               PARAM_BOUNDS['b'][0], 
                               PARAM_BOUNDS['b'][1])
    else:
        # Exploitation: use TPE sampling (default)
        chunk_size = trial.suggest_int('chunk_size', 
                                      PARAM_BOUNDS['chunk_size'][0], 
                                      PARAM_BOUNDS['chunk_size'][1])
        overlap = trial.suggest_int('overlap', 
                                   PARAM_BOUNDS['overlap'][0], 
                                   PARAM_BOUNDS['overlap'][1])
        k1 = trial.suggest_float('k1', 
                                PARAM_BOUNDS['k1'][0], 
                                PARAM_BOUNDS['k1'][1])
        b = trial.suggest_float('b', 
                               PARAM_BOUNDS['b'][0], 
                               PARAM_BOUNDS['b'][1])
    
    # Only keep the essential constraint
    if overlap >= chunk_size:
        return 0.0  # Invalid configuration
    
    # Round parameters for consistency
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    k1 = round(k1, 2)
    b = round(b, 2)
    
    # Evaluate configuration
    accuracy = evaluate_config(chunk_size, overlap, k1, b)
    
    return accuracy

def run_bayesian_optimization(n_trials: int = 50):
    """Run Bayesian Optimization for BM25 hyperparameters."""
    print("🚀 BM25 BAYESIAN OPTIMIZATION")
    print("=" * 50)
    print(f"📊 Parameter bounds:")
    for param, (min_val, max_val) in PARAM_BOUNDS.items():
        print(f"   {param:12}: [{min_val:4.1f}, {max_val:4.1f}]")
    print(f"🎯 Target trials: {n_trials}")
    print(f"🔍 Exploration: First 300 trials")
    print("-" * 50)
    
    # Load existing grid search results
    grid_results_file = CACHE_ROOT / "grid_search_results.json"
    if grid_results_file.exists():
        print("📂 Loading existing grid search results...")
        with open(grid_results_file, 'r') as f:
            existing_results = json.load(f)
        print(f"✅ Loaded {len(existing_results)} existing configurations")
    else:
        existing_results = []
        print("⚠️  No existing results found")
    
    # Create study with exploration-exploitation strategy
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=int(n_trials * EXPLORATION_RATIO)),
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Add existing results to study
    if existing_results:
        print("\n📈 Adding existing results to study...")
        for result in existing_results:
            # Create a trial with existing parameters
            trial = study.ask()
            trial.suggest_int('chunk_size', result['chunk_size'], result['chunk_size'])
            trial.suggest_int('overlap', result['overlap'], result['overlap'])
            trial.suggest_float('k1', result['k1'], result['k1'])
            trial.suggest_float('b', result['b'], result['b'])
            
            # Tell the study about this result
            study.tell(trial, result['accuracy'])
    
    # Initialize results tracking
    results_file = CACHE_ROOT / "bo_results.json"
    all_results = []
    
    # Run optimization with tqdm and real-time saving
    print(f"\n🔄 Starting optimization...")
    with tqdm(total=n_trials, desc="Bayesian Optimization", ncols=80) as pbar:
        def callback(study, trial):
            pbar.update(1)
            # Save result immediately
            if trial.value is not None:
                result = {
                    'trial': trial.number,
                    'chunk_size': trial.params['chunk_size'],
                    'overlap': trial.params['overlap'],
                    'k1': trial.params['k1'],
                    'b': trial.params['b'],
                    'accuracy': trial.value
                }
                all_results.append(result)
                
                # Update file after every trial
                sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
                with open(results_file, 'w') as f:
                    json.dump(sorted_results, f, indent=2)
                
                # Show current best
                best = sorted_results[0]
                pbar.set_postfix({
                    'Best': f"{best['accuracy']:.3f}",
                    'Size': best['chunk_size'],
                    'k1': best['k1'],
                    'b': best['b']
                })
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    
    # Final save
    sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    with open(results_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)
    
    # Print final results
    print(f"\n🏆 OPTIMIZATION COMPLETE")
    print("=" * 50)
    print(f"🎯 Best trial: {study.best_trial.number}")
    print(f"⭐ Best accuracy: {study.best_trial.value:.3f}")
    print(f"📁 Results saved: {results_file}")
    
    print(f"\n🏅 TOP 10 RESULTS")
    print("-" * 50)
    for i, result in enumerate(sorted_results[:10]):
        print(f"{i+1:2d}. {result['accuracy']:.3f} | "
              f"size={result['chunk_size']:3d} overlap={result['overlap']:2d} "
              f"k1={result['k1']:4.2f} b={result['b']:4.2f}")
    
    return study, sorted_results

def main():
    parser = argparse.ArgumentParser(description="BM25 Bayesian Optimization")
    parser.add_argument(
        "--trials", 
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )
    
    args = parser.parse_args()
    
    run_bayesian_optimization(args.trials)

if __name__ == "__main__":
    main() 