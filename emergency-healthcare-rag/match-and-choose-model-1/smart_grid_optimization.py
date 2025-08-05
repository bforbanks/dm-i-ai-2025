#!/usr/bin/env python3
"""
Smart Grid Search for Topic Model Optimization

A targeted, intelligent grid search that focuses on promising parameter regions
instead of exhaustive search. Uses knowledge from previous runs to guide search.

Expected to find 90%+ configurations in 1-2 hours vs 8+ hours for exhaustive search.
"""

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import itertools
import random

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

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
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())
CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# -----------------------------------------------------------------------
# SMART GRID SEARCH CONFIGURATION
# -----------------------------------------------------------------------

@dataclass
class SmartGridConfig:
    """Configuration for smart grid search optimization"""
    max_samples: int = 200
    use_condensed_topics: bool = True
    max_evaluations: int = 150  # Much more focused than exhaustive
    
    # PHASE 1: BM25-focused search (find sweet spots)
    phase1_chunk_sizes: List[int] = None
    phase1_overlap_ratios: List[float] = None
    
    # PHASE 2: Model/strategy focused search (on best BM25 configs)
    phase2_models: List[str] = None
    phase2_strategies: List[str] = None
    top_bm25_configs: int = 5
    
    # PHASE 3: Fine-tuning around best configs
    phase3_enabled: bool = True
    phase3_radius: int = 2  # Parameter variations around best
    
    def __post_init__(self):
        if self.phase1_chunk_sizes is None:
            # Focus on promising ranges based on typical good performance
            self.phase1_chunk_sizes = [80, 96, 112, 128, 144, 160]
            
        if self.phase1_overlap_ratios is None:
            # Focus on ratios that typically work well
            self.phase1_overlap_ratios = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25]
            
        if self.phase2_models is None:
            # Include the model that achieved 91% and related ones
            self.phase2_models = [
                'sentence-transformers/all-distilroberta-v1',  # The breakthrough model
                'sentence-transformers/all-mpnet-base-v2',     # Often performs well
                'sentence-transformers/all-MiniLM-L6-v2',     # Fast baseline
                'sentence-transformers/all-roberta-large-v1'  # High-capacity
            ]
            
        if self.phase2_strategies is None:
            # Focus on strategies that often work well, including linear_0.7
            self.phase2_strategies = [
                'bm25_only',      # Baseline
                'linear_0.5',     # Balanced
                'linear_0.7',     # The breakthrough strategy
                'linear_0.8',     # High semantic weight
                'rrf',            # Often competitive
                'semantic_only'   # Pure semantic baseline
            ]

# -----------------------------------------------------------------------
# UTILITY FUNCTIONS (same as before)
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
    topic_dir = CONDENSED_TOPIC_DIR if use_condensed else Path("data/topics")
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

def load_statements(max_samples: int = 200) -> List[Tuple[str, int]]:
    """Load evaluation statements"""
    statements = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        answer = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        statements.append((statement, answer["statement_topic"]))
    return statements[:max_samples]

def evaluate_configuration(chunk_size: int, overlap: int, model_name: str, strategy: str, 
                         statements: List, config: SmartGridConfig) -> Dict:
    """Evaluate a single configuration"""
    start_time = time.time()
    
    # Build indices
    bm25_data = build_bm25_index(chunk_size, overlap, config.use_condensed_topics)
    semantic_data = None
    
    if strategy != 'bm25_only':
        semantic_data = build_semantic_index(model_name, bm25_data, chunk_size, overlap, config.use_condensed_topics)
        if semantic_data is None:
            return {'error': f'Could not load {model_name}'}
    
    # Evaluate
    correct_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    reciprocal_ranks = []
    
    for stmt, true_topic in statements:
        predicted_topics = hybrid_search(stmt, bm25_data, semantic_data, strategy, top_k=5)
        
        # Top-k accuracy
        for k in range(1, 6):
            if true_topic in predicted_topics[:k]:
                correct_counts[k] += 1
        
        # MRR
        try:
            rank = predicted_topics.index(true_topic) + 1
            reciprocal_ranks.append(1.0 / rank)
        except (ValueError, IndexError):
            reciprocal_ranks.append(0.0)
    
    elapsed = time.time() - start_time
    total = len(statements)
    
    return {
        'chunk_size': chunk_size,
        'overlap': overlap,
        'overlap_ratio': overlap / chunk_size,
        'model': model_name,
        'strategy': strategy,
        'top1_accuracy': correct_counts[1] / total,
        'top2_accuracy': correct_counts[2] / total,
        'top3_accuracy': correct_counts[3] / total,
        'top4_accuracy': correct_counts[4] / total,
        'top5_accuracy': correct_counts[5] / total,
        'mrr': np.mean(reciprocal_ranks),
        'evaluation_time': elapsed,
        'time_per_query': elapsed / total
    }

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
# SMART GRID SEARCH OPTIMIZATION
# -----------------------------------------------------------------------

def run_smart_grid_optimization():
    """Run smart grid search optimization"""
    config = SmartGridConfig()
    
    print("🚀 SMART GRID SEARCH OPTIMIZATION")
    print("=" * 60)
    print(f"📋 Configuration:")
    print(f"   📊 Evaluation samples: {config.max_samples}")
    print(f"   🗂️  Topics: {'Condensed' if config.use_condensed_topics else 'Original'}")
    print(f"   🎯 Max evaluations: {config.max_evaluations}")
    print(f"   ⏱️  Estimated runtime: 1-2 hours")
    print(f"   💾 Auto-save: Every 10 evaluations")
    
    # Load statements
    statements = load_statements(config.max_samples)
    print(f"📚 Loaded {len(statements)} evaluation statements")
    
    all_results = []
    start_time = time.time()
    evaluation_count = 0
    
    try:
        # ===============================================================
        # PHASE 1: BM25 PARAMETER OPTIMIZATION
        # ===============================================================
        print(f"\n" + "🔨" * 60)
        print(f"🔨 PHASE 1: BM25 PARAMETER OPTIMIZATION")
        print(f"🔨" * 60)
        print(f"   🎯 Testing {len(config.phase1_chunk_sizes)} x {len(config.phase1_overlap_ratios)} = {len(config.phase1_chunk_sizes) * len(config.phase1_overlap_ratios)} BM25 configurations")
        
        bm25_results = []
        phase1_configs = list(itertools.product(config.phase1_chunk_sizes, config.phase1_overlap_ratios))
        
        for i, (chunk_size, overlap_ratio) in enumerate(phase1_configs, 1):
            overlap = int(chunk_size * overlap_ratio)
            if overlap >= chunk_size:  # Skip invalid configs
                continue
                
            print(f"\n🔨 BM25 Config {i}/{len(phase1_configs)}: chunk_size={chunk_size}, overlap={overlap}")
            
            result = evaluate_configuration(chunk_size, overlap, '', 'bm25_only', statements, config)
            if 'error' not in result:
                bm25_results.append(result)
                all_results.append(result)
                evaluation_count += 1
                
                print(f"   ✅ T1: {result['top1_accuracy']:.3f}, T3: {result['top3_accuracy']:.3f}, T5: {result['top5_accuracy']:.3f}")
                
                # Save progress
                if evaluation_count % 10 == 0:
                    save_progress(all_results, evaluation_count, config.max_evaluations)
        
        # Rank BM25 results
        bm25_results.sort(key=lambda x: x['top1_accuracy'], reverse=True)
        
        print(f"\n📊 PHASE 1 COMPLETE - Top {min(5, len(bm25_results))} BM25 configurations:")
        for i, result in enumerate(bm25_results[:5], 1):
            print(f"   #{i}: cs={result['chunk_size']}, ov={result['overlap']} → T1={result['top1_accuracy']:.3f}")
        
        # ===============================================================
        # PHASE 2: MODEL/STRATEGY OPTIMIZATION ON BEST BM25 CONFIGS
        # ===============================================================
        print(f"\n" + "🧠" * 60)
        print(f"🧠 PHASE 2: MODEL/STRATEGY OPTIMIZATION")
        print(f"🧠" * 60)
        
        # Select top BM25 configs for semantic fusion
        top_bm25 = bm25_results[:config.top_bm25_configs]
        phase2_total = len(top_bm25) * len(config.phase2_models) * (len(config.phase2_strategies) - 1)  # -1 for bm25_only
        print(f"   🎯 Testing top {len(top_bm25)} BM25 configs with {len(config.phase2_models)} models and {len(config.phase2_strategies)-1} semantic strategies")
        print(f"   📊 Total semantic configs: {phase2_total}")
        
        phase2_count = 0
        for bm25_config in top_bm25:
            chunk_size = bm25_config['chunk_size']
            overlap = bm25_config['overlap']
            
            print(f"\n🔧 BM25 Config: chunk_size={chunk_size}, overlap={overlap} (T1={bm25_config['top1_accuracy']:.3f})")
            
            for model in config.phase2_models:
                print(f"   🧠 Testing {model.split('/')[-1]}...")
                
                for strategy in config.phase2_strategies:
                    if strategy == 'bm25_only':  # Skip, already tested in Phase 1
                        continue
                        
                    phase2_count += 1
                    print(f"      🔍 Strategy {phase2_count}/{phase2_total}: {strategy}")
                    
                    result = evaluate_configuration(chunk_size, overlap, model, strategy, statements, config)
                    if 'error' not in result:
                        all_results.append(result)
                        evaluation_count += 1
                        
                        print(f"         ✅ T1: {result['top1_accuracy']:.3f}, T3: {result['top3_accuracy']:.3f}, MRR: {result['mrr']:.3f}")
                        
                        # Save progress
                        if evaluation_count % 10 == 0:
                            save_progress(all_results, evaluation_count, config.max_evaluations)
                    else:
                        print(f"         ❌ {result['error']}")
        
        # ===============================================================
        # PHASE 3: FINE-TUNING AROUND BEST CONFIGURATIONS
        # ===============================================================
        if config.phase3_enabled and evaluation_count < config.max_evaluations:
            print(f"\n" + "🎯" * 60)
            print(f"🎯 PHASE 3: FINE-TUNING AROUND BEST CONFIGURATIONS")
            print(f"🎯" * 60)
            
            # Find best configurations so far
            semantic_results = [r for r in all_results if r['strategy'] != 'bm25_only']
            if semantic_results:
                semantic_results.sort(key=lambda x: x['top1_accuracy'], reverse=True)
                best_configs = semantic_results[:3]  # Top 3 for fine-tuning
                
                print(f"   🏆 Fine-tuning around top {len(best_configs)} configurations:")
                for i, cfg in enumerate(best_configs, 1):
                    print(f"      #{i}: {cfg['model'].split('/')[-1]} + {cfg['strategy']} → T1={cfg['top1_accuracy']:.3f}")
                
                # Generate fine-tuning variations
                fine_tune_configs = []
                for cfg in best_configs:
                    # Chunk size variations
                    for delta in [-16, -8, 8, 16]:
                        new_chunk = cfg['chunk_size'] + delta
                        if 64 <= new_chunk <= 192:
                            fine_tune_configs.append({
                                'chunk_size': new_chunk,
                                'overlap': int(new_chunk * cfg['overlap_ratio']),
                                'model': cfg['model'],
                                'strategy': cfg['strategy'],
                                'source': f"chunk_variation_of_{cfg['chunk_size']}"
                            })
                    
                    # Strategy variations (if linear)
                    if cfg['strategy'].startswith('linear_'):
                        current_weight = float(cfg['strategy'].split('_')[1])
                        for delta in [-0.1, -0.05, 0.05, 0.1]:
                            new_weight = current_weight + delta
                            if 0.1 <= new_weight <= 0.9:
                                fine_tune_configs.append({
                                    'chunk_size': cfg['chunk_size'],
                                    'overlap': cfg['overlap'],
                                    'model': cfg['model'],
                                    'strategy': f"linear_{new_weight:.2f}",
                                    'source': f"strategy_variation_of_{cfg['strategy']}"
                                })
                
                # Limit fine-tuning to remaining budget
                remaining_budget = config.max_evaluations - evaluation_count
                fine_tune_configs = fine_tune_configs[:remaining_budget]
                
                print(f"   🔬 Testing {len(fine_tune_configs)} fine-tuning variations")
                
                for i, ft_config in enumerate(fine_tune_configs, 1):
                    print(f"      🎯 Fine-tune {i}/{len(fine_tune_configs)}: {ft_config['source']}")
                    
                    result = evaluate_configuration(
                        ft_config['chunk_size'], ft_config['overlap'],
                        ft_config['model'], ft_config['strategy'],
                        statements, config
                    )
                    
                    if 'error' not in result:
                        result['fine_tuning_source'] = ft_config['source']
                        all_results.append(result)
                        evaluation_count += 1
                        
                        print(f"         ✅ T1: {result['top1_accuracy']:.3f}, T3: {result['top3_accuracy']:.3f}")
        
        # ===============================================================
        # FINAL RESULTS
        # ===============================================================
        elapsed = time.time() - start_time
        
        # Sort all results by accuracy
        all_results.sort(key=lambda x: x['top1_accuracy'], reverse=True)
        
        print(f"\n" + "🎉" * 60)
        print(f"🎉 SMART GRID OPTIMIZATION COMPLETE!")
        print(f"🎉" * 60)
        print(f"⏱️  Total time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
        print(f"🔍 Total evaluations: {evaluation_count}")
        print(f"🚀 Best accuracy: {all_results[0]['top1_accuracy']:.3f}")
        
        print(f"\n🏆 TOP 10 CONFIGURATIONS:")
        for i, result in enumerate(all_results[:10], 1):
            model_short = result['model'].split('/')[-1] if result['model'] else 'BM25-only'
            print(f"   #{i}: {model_short} + {result['strategy']} → T1={result['top1_accuracy']:.3f}")
            print(f"       (cs={result['chunk_size']}, ov={result['overlap']}, ratio={result['overlap_ratio']:.3f})")
        
        # Save final results
        final_results = {
            'smart_grid_optimization': {
                'config': config.__dict__,
                'total_evaluations': evaluation_count,
                'optimization_time_hours': elapsed / 3600,
                'best_configuration': all_results[0] if all_results else None
            },
            'all_results': all_results,
            'phase_breakdown': {
                'phase1_bm25_configs': len([r for r in all_results if r['strategy'] == 'bm25_only']),
                'phase2_semantic_configs': len([r for r in all_results if r['strategy'] != 'bm25_only' and 'fine_tuning_source' not in r]),
                'phase3_fine_tuning_configs': len([r for r in all_results if 'fine_tuning_source' in r])
            }
        }
        
        final_file = "smart_grid_optimization_results.json"
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n💾 Final results saved: {final_file}")
        
        return all_results[0] if all_results else None
        
    except KeyboardInterrupt:
        print(f"\n🛑 OPTIMIZATION INTERRUPTED")
        save_progress(all_results, evaluation_count, config.max_evaluations, interrupted=True)
        return None

def save_progress(all_results: List[Dict], completed: int, total: int, interrupted: bool = False):
    """Save optimization progress"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"smart_grid_progress{'_interrupted' if interrupted else ''}_{timestamp}.json"
    
    progress_data = {
        'progress': {
            'completed_evaluations': completed,
            'total_evaluations': total,
            'progress_percentage': (completed / total) * 100,
            'interrupted': interrupted
        },
        'current_best': max(all_results, key=lambda x: x['top1_accuracy']) if all_results else None,
        'all_results': all_results
    }
    
    with open(filename, 'w') as f:
        json.dump(progress_data, f, indent=2, default=str)
    
    print(f"   💾 Progress saved: {filename}")

if __name__ == "__main__":
    run_smart_grid_optimization()