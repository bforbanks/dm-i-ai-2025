#!/usr/bin/env python3
"""
Bayesian Optimization for Topic Model Hyperparameter Tuning

Uses Gaussian Process-based optimization to efficiently find optimal configurations
for hybrid BM25 + semantic search. Much more efficient than grid search.

Expected to find 90%+ configurations in 2-3 hours vs 8+ hours for grid search.
"""

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

try:
    import skopt
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
except ImportError:
    print("❌ Missing scikit-optimize. Install with: pip install scikit-optimize")
    exit(1)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from rank_bm25 import BM25Okapi
    import tqdm
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("📦 Install: pip install sentence-transformers scikit-learn rank_bm25 tqdm")
    exit(1)

# -----------------------------------------------------------------------
# PATHS AND CONSTANTS
# -----------------------------------------------------------------------

CONDENSED_TOPIC_DIR = Path("data/condensed_topics")
ORIGINAL_TOPIC_DIR = Path("data/topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())
CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# BAYESIAN OPTIMIZATION CONFIGURATION
# -----------------------------------------------------------------------

@dataclass
class BayesianOptConfig:
    """Configuration for Bayesian optimization"""
    n_calls: int = 100  # Number of optimization iterations
    n_initial_points: int = 15  # Random initial points before GP kicks in
    acq_func: str = 'EI'  # Acquisition function (Expected Improvement)
    n_jobs: int = 1  # Parallel evaluations (keep 1 for sequential)
    random_state: int = 42
    max_samples: int = 200  # Evaluation samples
    use_condensed_topics: bool = True
    
    # Search space bounds
    chunk_size_min: int = 64
    chunk_size_max: int = 192
    overlap_ratio_min: float = 0.0
    overlap_ratio_max: float = 0.35
    
    def __post_init__(self):
        # Define optimization search space
        self.search_space = [
            Integer(self.chunk_size_min, self.chunk_size_max, name='chunk_size'),
            Real(self.overlap_ratio_min, self.overlap_ratio_max, name='overlap_ratio'),
            Categorical([
                'sentence-transformers/all-MiniLM-L6-v2',
                'sentence-transformers/all-mpnet-base-v2', 
                'sentence-transformers/all-distilroberta-v1',
                'sentence-transformers/all-roberta-large-v1'
            ], name='model'),
            Categorical([
                'bm25_only', 'semantic_only', 
                'linear_0.3', 'linear_0.5', 'linear_0.7', 'linear_0.8',
                'rrf'
            ], name='strategy')
        ]

# Global configuration
config = BayesianOptConfig()

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

def build_bm25_index(chunk_size: int, overlap: int, use_condensed: bool = True) -> Dict:
    """Build BM25 index with caching"""
    topic_dir = CONDENSED_TOPIC_DIR if use_condensed else ORIGINAL_TOPIC_DIR
    cache_key = f"{'condensed' if use_condensed else 'original'}_{chunk_size}_{overlap}"
    cache_path = CACHE_ROOT / f"bm25_index_{cache_key}.pkl"
    
    if cache_path.exists():
        try:
            data = pickle.loads(cache_path.read_bytes())
            if 'chunks' in data and 'topics' in data and 'bm25' in data:
                return data
        except:
            cache_path.unlink()
    
    chunks = []
    topics = []
    
    for md_file in topic_dir.rglob("*.md"):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP[topic_name]
        words = md_file.read_text(encoding="utf-8").split()
        
        for w_chunk in chunk_words(words, chunk_size, overlap):
            if len(w_chunk) < 10:
                continue
            chunk_text = " ".join(w_chunk)
            chunks.append(chunk_text)
            topics.append(topic_id)
    
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    data = {'chunks': chunks, 'topics': topics, 'bm25': bm25}
    cache_path.write_bytes(pickle.dumps(data))
    return data

def build_semantic_index(model_name: str, bm25_data: Dict, chunk_size: int, overlap: int, use_condensed: bool = True) -> Optional[Dict]:
    """Build semantic embeddings with caching"""
    cache_key = f"{model_name.replace('/', '_')}_{'condensed' if use_condensed else 'original'}_{chunk_size}_{overlap}"
    cache_path = CACHE_ROOT / f"semantic_index_{cache_key}.pkl"
    
    if cache_path.exists():
        try:
            data = pickle.loads(cache_path.read_bytes())
            if 'embeddings' in data:
                return data
        except:
            cache_path.unlink()
    
    try:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(bm25_data['chunks'], show_progress_bar=False, batch_size=32)
        
        data = {'embeddings': embeddings, 'model_name': model_name}
        cache_path.write_bytes(pickle.dumps(data))
        return data
    except Exception as e:
        print(f"⚠️  Failed to load {model_name}: {e}")
        return None

def load_statements() -> List[Tuple[str, int]]:
    """Load evaluation statements"""
    statements = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        answer = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        statements.append((statement, answer["statement_topic"]))
    return statements[:config.max_samples]

def hybrid_search(query: str, bm25_data: Dict, semantic_data: Optional[Dict], strategy: str, top_k: int = 5) -> List[int]:
    """Perform hybrid search and return topic rankings"""
    chunks = bm25_data['chunks']
    topics = bm25_data['topics']
    
    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25_data['bm25'].get_scores(tokenized_query)
    
    if strategy == 'bm25_only':
        final_scores = bm25_scores
    elif strategy == 'semantic_only':
        if semantic_data is None:
            return []
        model = SentenceTransformer(semantic_data['model_name'])
        query_embedding = model.encode([query])
        final_scores = cosine_similarity(query_embedding, semantic_data['embeddings'])[0]
    elif strategy.startswith('linear_'):
        if semantic_data is None:
            return []
        semantic_weight = float(strategy.split('_')[1])
        bm25_weight = 1.0 - semantic_weight
        
        model = SentenceTransformer(semantic_data['model_name'])
        query_embedding = model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, semantic_data['embeddings'])[0]
        
        # Normalize scores
        bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        final_scores = semantic_weight * semantic_norm + bm25_weight * bm25_norm
    elif strategy == 'rrf':
        if semantic_data is None:
            return []
        model = SentenceTransformer(semantic_data['model_name'])
        query_embedding = model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, semantic_data['embeddings'])[0]
        
        # Reciprocal Rank Fusion
        bm25_ranks = np.argsort(np.argsort(bm25_scores)[::-1])
        semantic_ranks = np.argsort(np.argsort(semantic_scores)[::-1])
        
        rrf_scores = 1.0 / (60 + bm25_ranks) + 1.0 / (60 + semantic_ranks)
        final_scores = rrf_scores
    else:
        return []
    
    # Get top-k topic predictions
    top_indices = np.argsort(final_scores)[-top_k:][::-1]
    return [topics[idx] for idx in top_indices]

# -----------------------------------------------------------------------
# BAYESIAN OPTIMIZATION OBJECTIVE FUNCTION
# -----------------------------------------------------------------------

# Global variables for optimization
statements_cache = None
call_counter = 0
best_score = 0.0
all_results = []

@use_named_args(config.search_space)
def objective_function(chunk_size: int, overlap_ratio: float, model: str, strategy: str) -> float:
    """
    Objective function for Bayesian optimization
    Returns: negative top-1 accuracy (since skopt minimizes)
    """
    global statements_cache, call_counter, best_score, all_results
    
    call_counter += 1
    overlap = int(chunk_size * overlap_ratio)
    
    print(f"\n🔬 EVALUATION {call_counter}/{config.n_calls}")
    print(f"   📊 Testing: {model.split('/')[-1]} + {strategy}")
    print(f"   ⚙️  BM25: chunk_size={chunk_size}, overlap={overlap} (ratio={overlap_ratio:.3f})")
    
    start_time = time.time()
    
    try:
        # Build indices
        bm25_data = build_bm25_index(chunk_size, overlap, config.use_condensed_topics)
        semantic_data = build_semantic_index(model, bm25_data, chunk_size, overlap, config.use_condensed_topics)
        
        if semantic_data is None and strategy != 'bm25_only':
            print(f"   ❌ Skipping - could not load {model}")
            return 1.0  # Return poor score
        
        # Load statements if not cached
        if statements_cache is None:
            statements_cache = load_statements()
        
        # Evaluate
        correct = 0
        for stmt, true_topic in statements_cache:
            predicted_topics = hybrid_search(stmt, bm25_data, semantic_data, strategy, top_k=5)
            if predicted_topics and predicted_topics[0] == true_topic:
                correct += 1
        
        accuracy = correct / len(statements_cache)
        elapsed = time.time() - start_time
        
        # Store result
        result = {
            'call_id': call_counter,
            'chunk_size': chunk_size,
            'overlap': overlap,
            'overlap_ratio': overlap_ratio,
            'model': model,
            'strategy': strategy,
            'top1_accuracy': accuracy,
            'evaluation_time': elapsed,
            'timestamp': time.time()
        }
        all_results.append(result)
        
        # Update best
        if accuracy > best_score:
            best_score = accuracy
            print(f"   🚀 NEW BEST: {accuracy:.3f} ({correct}/{len(statements_cache)}) | {elapsed:.1f}s")
            
            # Save best configuration immediately
            best_config_file = f"bayesian_best_config.json"
            with open(best_config_file, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(f"   ✅ Result: {accuracy:.3f} ({correct}/{len(statements_cache)}) | {elapsed:.1f}s")
        
        # Save progress every 5 evaluations
        if call_counter % 5 == 0:
            progress_file = f"bayesian_progress_{call_counter}.json"
            with open(progress_file, 'w') as f:
                json.dump({
                    'progress': {
                        'completed_calls': call_counter,
                        'total_calls': config.n_calls,
                        'best_accuracy': best_score,
                        'current_best_config': max(all_results, key=lambda x: x['top1_accuracy']) if all_results else None
                    },
                    'all_results': all_results
                }, f, indent=2)
            print(f"   💾 Progress saved: {progress_file}")
        
        # Return negative accuracy (skopt minimizes)
        return -accuracy
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 1.0  # Return poor score for failed evaluations

# -----------------------------------------------------------------------
# MAIN OPTIMIZATION FUNCTION
# -----------------------------------------------------------------------

def run_bayesian_optimization():
    """Run Bayesian optimization for topic model hyperparameters"""
    
    print("🚀 BAYESIAN OPTIMIZATION FOR TOPIC MODEL")
    print("=" * 60)
    print(f"📋 Configuration:")
    print(f"   🔍 Total evaluations: {config.n_calls}")
    print(f"   🎲 Initial random points: {config.n_initial_points}")
    print(f"   🧠 Acquisition function: {config.acq_func}")
    print(f"   📊 Evaluation samples: {config.max_samples}")
    print(f"   🗂️  Topics: {'Condensed' if config.use_condensed_topics else 'Original'}")
    print(f"   📦 Search space:")
    print(f"      • Chunk size: {config.chunk_size_min}-{config.chunk_size_max}")
    print(f"      • Overlap ratio: {config.overlap_ratio_min:.2f}-{config.overlap_ratio_max:.2f}")
    print(f"      • Models: 4 embedding models")
    print(f"      • Strategies: 7 fusion strategies")
    print(f"   ⏱️  Estimated runtime: 2-4 hours")
    print(f"   💾 Auto-save: Every 5 evaluations + best configs")
    
    # Initialize optimization
    start_time = time.time()
    
    try:
        print(f"\n🎯 STARTING BAYESIAN OPTIMIZATION...")
        print(f"   🔍 Phase 1: {config.n_initial_points} random explorations")
        print(f"   🧠 Phase 2: {config.n_calls - config.n_initial_points} Gaussian Process guided searches")
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective_function,
            dimensions=config.search_space,
            n_calls=config.n_calls,
            n_initial_points=config.n_initial_points,
            acq_func=config.acq_func,
            random_state=config.random_state,
            verbose=False  # We handle our own progress printing
        )
        
        elapsed = time.time() - start_time
        
        # Extract best configuration
        best_params = result.x
        best_accuracy = -result.fun  # Convert back from negative
        
        best_config = {
            'chunk_size': best_params[0],
            'overlap_ratio': best_params[1],
            'overlap': int(best_params[0] * best_params[1]),
            'model': best_params[2],
            'strategy': best_params[3],
            'top1_accuracy': best_accuracy,
            'optimization_time': elapsed
        }
        
        print(f"\n" + "🎉" * 60)
        print(f"🎉 BAYESIAN OPTIMIZATION COMPLETE!")
        print(f"🎉" * 60)
        print(f"⏱️  Total time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
        print(f"🔍 Total evaluations: {len(all_results)}")
        print(f"🚀 Best accuracy: {best_accuracy:.3f}")
        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"   🧠 Model: {best_config['model']}")
        print(f"   🔍 Strategy: {best_config['strategy']}")
        print(f"   ⚙️  BM25: chunk_size={best_config['chunk_size']}, overlap={best_config['overlap']}")
        print(f"   📊 Overlap ratio: {best_config['overlap_ratio']:.3f}")
        
        # Save final results
        final_results = {
            'bayesian_optimization': {
                'config': config.__dict__,
                'best_configuration': best_config,
                'optimization_result': {
                    'best_params': best_params,
                    'best_score': best_accuracy,
                    'n_calls': len(result.x_iters),
                    'total_time': elapsed
                },
                'convergence_data': {
                    'x_iters': result.x_iters,
                    'func_vals': result.func_vals
                }
            },
            'all_evaluations': all_results,
            'summary': {
                'total_evaluations': len(all_results),
                'best_accuracy': best_accuracy,
                'optimization_time_hours': elapsed / 3600,
                'evaluations_per_hour': len(all_results) / (elapsed / 3600)
            }
        }
        
        final_file = "bayesian_optimization_results.json"
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n💾 Final results saved: {final_file}")
        print(f"📈 Convergence plot data included for analysis")
        
        # Top 5 configurations
        top_configs = sorted(all_results, key=lambda x: x['top1_accuracy'], reverse=True)[:5]
        print(f"\n🏅 TOP 5 CONFIGURATIONS:")
        for i, cfg in enumerate(top_configs, 1):
            print(f"   #{i}: {cfg['model'].split('/')[-1]} + {cfg['strategy']} → {cfg['top1_accuracy']:.3f}")
            print(f"       (cs={cfg['chunk_size']}, ov={cfg['overlap']}, ratio={cfg['overlap_ratio']:.3f})")
        
        return best_config
        
    except KeyboardInterrupt:
        print(f"\n🛑 OPTIMIZATION INTERRUPTED")
        print(f"💾 Saving current progress...")
        
        interrupt_file = f"bayesian_optimization_interrupted.json"
        with open(interrupt_file, 'w') as f:
            json.dump({
                'interrupted_at': call_counter,
                'best_score': best_score,
                'all_results': all_results,
                'interruption_time': time.time()
            }, f, indent=2)
        
        print(f"📂 Progress saved: {interrupt_file}")
        print(f"🔍 Completed {call_counter}/{config.n_calls} evaluations")
        if best_score > 0:
            print(f"🚀 Best found so far: {best_score:.3f}")
        
        return None

if __name__ == "__main__":
    run_bayesian_optimization()