#!/usr/bin/env python3
"""
Optimize topic model with hybrid BM25 + semantic search approaches
Tests multiple embedding models and fusion strategies to find complementary models
"""

import os
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass

# Suppress transformer warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

# Core dependencies
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Cloud-friendly progress bars
try:
    from tqdm import tqdm
    # Configure tqdm for cloud environments
    tqdm.monitor_interval = 0  # Disable monitoring thread
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    def tqdm(iterable, desc="", leave=True, **kwargs):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:  # Print every 5%
                print(f"{desc}: {i+1}/{total} ({100*(i+1)/total:.0f}%)")
            yield item

# ----------------------------------------------------------------------- 
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs"""
    # Models to test (will be downloaded automatically)
    embedding_models: List[str] = None
    
    # Fusion strategies to test
    fusion_strategies: List[str] = None
    
    # BM25 optimization parameters
    chunk_sizes: List[int] = None
    overlap_ratios: List[float] = None  # As fraction of chunk size
    
    # Topics to use
    use_condensed_topics: bool = True  # True for condensed, False for original
    
    # Search parameters
    top_k: int = 10
    
    # Evaluation parameters
    max_samples: int = 200  # Use full train set for comprehensive evaluation
    cache_embeddings: bool = True
    save_detailed_results: bool = True
    optimize_bm25: bool = True  # Whether to optimize BM25 parameters
    
    # SMART OPTIMIZATION PARAMETERS
    fast_mode: bool = True  # Enable hierarchical optimization for speed
    top_bm25_configs: int = 3  # Only test top N BM25 configs for semantic fusion
    sample_strategies: bool = True  # Sample strategies instead of exhaustive test
    target_runtime_hours: float = 1.5  # Target total runtime
    
    # ADAPTIVE EXPLORATION PARAMETERS
    enable_adaptive_exploration: bool = True  # Detect promising paths and explore deeper
    breakthrough_threshold: float = 0.905  # Accuracy threshold to trigger deeper exploration
    exploration_radius: int = 2  # How many parameter variations to test around breakthroughs
    max_exploration_configs: int = 50  # Max additional configs per breakthrough
    
    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = [
                "sentence-transformers/all-MiniLM-L6-v2",       # Fast, lightweight (384d)
                "sentence-transformers/all-mpnet-base-v2",      # Best general performance (768d)
                "sentence-transformers/all-distilroberta-v1",   # Different architecture (768d)
            ]
        
        if self.fusion_strategies is None:
            if self.fast_mode and self.sample_strategies:
                # Smart strategy sampling for fast mode - focus on most promising
                self.fusion_strategies = [
                    "bm25_only",           # Always include baseline
                    "semantic_only",       # Pure semantic baseline
                    "linear_0.5",          # Balanced fusion (often best)
                    "rrf",                 # RRF often outperforms linear
                ]
            else:
                # Full strategy testing
                self.fusion_strategies = [
                    "bm25_only",           # Baseline
                    "semantic_only",       # Semantic baseline
                    "linear_0.3",          # 0.3*BM25 + 0.7*semantic
                    "linear_0.5",          # 0.5*BM25 + 0.5*semantic  
                    "linear_0.7",          # 0.7*BM25 + 0.3*semantic
                    "rrf",                 # Reciprocal Rank Fusion
                    "adaptive"             # Adaptive weighting
                ]
        
        if self.chunk_sizes is None:
            if self.optimize_bm25:
                # DEEP BM25 optimization - test many configs since BM25 is fast
                self.chunk_sizes = [64, 80, 96, 112, 128, 144, 160, 176, 192]
                self.overlap_ratios = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            else:
                # Use known best config (Elias: 89.5% with chunk=128, overlap=12)
                self.chunk_sizes = [128]
                self.overlap_ratios = [0.094]  # 12/128 ≈ 0.094

# Global paths
CONDENSED_TOPIC_DIR = Path("data/condensed_topics")
ORIGINAL_TOPIC_DIR = Path("data/topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)

# Load topic mapping
with open('data/topics.json', 'r') as f:
    TOPIC_MAP = json.load(f)

# -----------------------------------------------------------------------
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------

def chunk_words(words: List[str], size: int, overlap: int) -> List[str]:
    """Generate word chunks with overlap"""
    step = max(1, size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        chunk = words[i:i + size]
        if len(chunk) < 10:  # Skip very small fragments
            continue
        chunks.append(" ".join(chunk))
        if i + size >= len(words):
            break
    return chunks

def load_statements() -> List[Tuple[str, int]]:
    """Load all training statements with their true topics"""
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive evaluation metrics"""
    total = len(results)
    top1_correct = sum(1 for r in results if r['rank_correct'] == 1)
    top2_correct = sum(1 for r in results if r['rank_correct'] <= 2)
    top3_correct = sum(1 for r in results if r['rank_correct'] <= 3)
    top4_correct = sum(1 for r in results if r['rank_correct'] <= 4)
    top5_correct = sum(1 for r in results if r['rank_correct'] <= 5)
    
    # Mean Reciprocal Rank
    mrr = np.mean([1.0 / r['rank_correct'] if r['rank_correct'] > 0 else 0.0 for r in results])
    
    # Average rank of correct answer
    valid_ranks = [r['rank_correct'] for r in results if r['rank_correct'] > 0]
    avg_rank = np.mean(valid_ranks) if valid_ranks else float('inf')
    
    # Score separation analysis (when 1st pick is correct)
    correct_first = [r for r in results if r['rank_correct'] == 1]
    score_separations = []
    if correct_first:
        for r in correct_first:
            if len(r['top_scores']) >= 2:
                sep = r['top_scores'][0] - r['top_scores'][1]
                score_separations.append(sep)
    
    avg_separation = np.mean(score_separations) if score_separations else 0.0
    
    # Confidence analysis (gap between 1st and 2nd pick)
    gaps = []
    for r in results:
        if len(r['top_scores']) >= 2:
            gap = r['top_scores'][0] - r['top_scores'][1]
            gaps.append(gap)
    
    return {
        'total_samples': total,
        'top1_accuracy': top1_correct / total,
        'top2_accuracy': top2_correct / total,
        'top3_accuracy': top3_correct / total,
        'top4_accuracy': top4_correct / total,  
        'top5_accuracy': top5_correct / total,
        'mrr': mrr,
        'avg_rank': avg_rank,
        'avg_separation_when_correct': avg_separation,
        'score_gaps': {
            'mean': np.mean(gaps) if gaps else 0.0,
            'std': np.std(gaps) if gaps else 0.0,
            'percentiles': {
                'p25': np.percentile(gaps, 25) if gaps else 0.0,
                'p50': np.percentile(gaps, 50) if gaps else 0.0,
                'p75': np.percentile(gaps, 75) if gaps else 0.0,
                'p90': np.percentile(gaps, 90) if gaps else 0.0
            }
        }
    }

# -----------------------------------------------------------------------
# BM25 INDEX BUILDING
# -----------------------------------------------------------------------

def build_bm25_index(chunk_size: int, overlap: int, use_condensed_topics: bool = True) -> Dict:
    """Build BM25 index from topics"""
    topic_type = "condensed" if use_condensed_topics else "original"
    cache_path = CACHE_ROOT / f"bm25_index_{topic_type}_{chunk_size}_{overlap}.pkl"
    
    if cache_path.exists():
        print(f"   📚 Loading cached BM25 index ({topic_type}, {chunk_size}, {overlap})...")
        try:
            data = pickle.loads(cache_path.read_bytes())
            # Check if it has the expected structure
            if 'chunks' in data and 'topics' in data:
                print(f"   ✅ Cache loaded successfully")
                return data
            else:
                print(f"   ⚠️ Old cache format detected, rebuilding...")
                cache_path.unlink()  # Delete old cache
        except Exception as e:
            print(f"   ⚠️ Cache loading failed: {e}, rebuilding...")
            cache_path.unlink()  # Delete corrupted cache

    topic_dir = CONDENSED_TOPIC_DIR if use_condensed_topics else ORIGINAL_TOPIC_DIR
    print(f"   🔨 Building BM25 index — {topic_type}_topics size={chunk_size} overlap={overlap}")
    
    chunks = []
    topics = []
    topic_names = []
    
    for md_file in tqdm(topic_dir.rglob("*.md"), desc=f"Processing {topic_type} topics", leave=False):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP.get(topic_name, -1)
        
        if topic_id == -1:
            continue
            
        words = md_file.read_text(encoding="utf-8").split()
        for chunk_text in chunk_words(words, chunk_size, overlap):
            chunks.append(chunk_text)
            topics.append(topic_id)
            topic_names.append(topic_name)

    # Build BM25 index
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    data = {
        'topics': topics,
        'chunks': chunks,
        'topic_names': topic_names,
        'bm25': bm25,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'use_condensed_topics': use_condensed_topics
    }
    
    cache_path.write_bytes(pickle.dumps(data))
    print(f"💾 Cached BM25 index to {cache_path}")
    return data

# -----------------------------------------------------------------------
# SEMANTIC EMBEDDING BUILDING
# -----------------------------------------------------------------------

def build_semantic_index(model_name: str, bm25_data: Dict, cache_embeddings: bool = True) -> Dict:
    """Build semantic embeddings for all chunks"""
    model_slug = model_name.replace("/", "_").replace("-", "_")
    chunk_size = bm25_data['chunk_size']
    overlap = bm25_data['overlap']
    topic_type = "condensed" if bm25_data['use_condensed_topics'] else "original"
    cache_path = CACHE_ROOT / f"semantic_index_{model_slug}_{topic_type}_{chunk_size}_{overlap}.pkl"
    
    if cache_path.exists() and cache_embeddings:
        print(f"      📚 Loading cached semantic index for {model_name.split('/')[-1]}...")
        try:
            data = pickle.loads(cache_path.read_bytes())
            # Check if it has the expected structure
            if 'embeddings' in data and 'model_name' in data:
                print(f"      ✅ Semantic cache loaded successfully")
                return data
            else:
                print(f"      ⚠️ Old cache format detected, rebuilding...")
                cache_path.unlink()  # Delete old cache
        except Exception as e:
            print(f"      ⚠️ Cache loading failed: {e}, rebuilding...")
            cache_path.unlink()  # Delete corrupted cache

    print(f"      🤖 Building semantic index for {model_name.split('/')[-1]}...")
    
    # Load model
    try:
        model = SentenceTransformer(model_name)
        print(f"✅ Loaded {model_name}")
    except Exception as e:
        print(f"❌ Failed to load {model_name}: {e}")
        return None
    
    # Generate embeddings in batches
    chunks = bm25_data['chunks']
    batch_size = 32
    embeddings = []
    
    print(f"🔢 Generating embeddings for {len(chunks)} chunks...")
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches", leave=False):
        batch = chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(embeddings)
    
    data = {
        'embeddings': all_embeddings,
        'model_name': model_name,
        'topics': bm25_data['topics'],
        'chunks': bm25_data['chunks'],
        'topic_names': bm25_data['topic_names'],
        'bm25_data': bm25_data
    }
    
    if cache_embeddings:
        cache_path.write_bytes(pickle.dumps(data))
        print(f"💾 Cached semantic index to {cache_path}")
    
    return data

# -----------------------------------------------------------------------
# SEARCH STRATEGIES
# -----------------------------------------------------------------------

class HybridSearcher:
    """Hybrid search combining BM25 and semantic similarity"""
    
    def __init__(self, bm25_data: Dict, semantic_data: Optional[Dict] = None):
        self.bm25_data = bm25_data
        self.semantic_data = semantic_data
        self.semantic_model = None
        
        # Load semantic model if needed
        if semantic_data and 'model_name' in semantic_data:
            try:
                self.semantic_model = SentenceTransformer(semantic_data['model_name'])
            except:
                print(f"⚠️ Could not load semantic model {semantic_data['model_name']}")
    
    def search_bm25_only(self, query: str, top_k: int = 10) -> List[Dict]:
        """BM25-only search"""
        tokenized_query = query.lower().split()
        scores = self.bm25_data['bm25'].get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return self._format_results(top_indices, scores)
    
    def search_semantic_only(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic-only search"""
        if not self.semantic_model or not self.semantic_data:
            return []
        
        # Encode query
        query_embedding = self.semantic_model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.semantic_data['embeddings'])[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return self._format_results(top_indices, similarities)
    
    def search_linear_fusion(self, query: str, bm25_weight: float = 0.5, top_k: int = 10) -> List[Dict]:
        """Linear combination of BM25 and semantic scores"""
        if not self.semantic_model or not self.semantic_data:
            return self.search_bm25_only(query, top_k)
        
        # Get BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_data['bm25'].get_scores(tokenized_query)
        
        # Get semantic scores
        query_embedding = self.semantic_model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.semantic_data['embeddings'])[0]
        
        # Normalize scores to [0, 1]
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        semantic_scores_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
        
        # Linear combination
        combined_scores = bm25_weight * bm25_scores_norm + (1 - bm25_weight) * semantic_scores_norm
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        return self._format_results(top_indices, combined_scores)
    
    def search_rrf(self, query: str, top_k: int = 10, k_param: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion of BM25 and semantic search"""
        if not self.semantic_model or not self.semantic_data:
            return self.search_bm25_only(query, top_k)
        
        # Get BM25 rankings
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_data['bm25'].get_scores(tokenized_query)
        bm25_rankings = np.argsort(bm25_scores)[::-1]
        
        # Get semantic rankings
        query_embedding = self.semantic_model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.semantic_data['embeddings'])[0]
        semantic_rankings = np.argsort(semantic_scores)[::-1]
        
        # Calculate RRF scores
        rrf_scores = np.zeros(len(bm25_scores))
        
        # Add BM25 contributions
        for rank, idx in enumerate(bm25_rankings):
            rrf_scores[idx] += 1.0 / (k_param + rank + 1)
        
        # Add semantic contributions
        for rank, idx in enumerate(semantic_rankings):
            rrf_scores[idx] += 1.0 / (k_param + rank + 1)
        
        # Get top-k indices
        top_indices = np.argsort(rrf_scores)[-top_k:][::-1]
        
        return self._format_results(top_indices, rrf_scores)
    
    def search_adaptive(self, query: str, top_k: int = 10) -> List[Dict]:
        """Adaptive weighting based on query characteristics"""
        if not self.semantic_model or not self.semantic_data:
            return self.search_bm25_only(query, top_k)
        
        # Analyze query characteristics
        query_length = len(query.split())
        has_numbers = any(char.isdigit() for char in query)
        has_medical_terms = any(term in query.lower() for term in ['syndrome', 'disease', 'symptom', 'treatment', 'diagnosis'])
        
        # Adaptive weighting heuristics
        if has_numbers or query_length < 5:
            # Short queries or queries with numbers -> favor BM25
            bm25_weight = 0.8
        elif has_medical_terms and query_length > 10:
            # Long medical queries -> favor semantic
            bm25_weight = 0.3
        else:
            # Balanced approach
            bm25_weight = 0.5
        
        return self.search_linear_fusion(query, bm25_weight, top_k)
    
    def _format_results(self, indices: np.ndarray, scores: np.ndarray) -> List[Dict]:
        """Format search results consistently"""
        results = []
        for idx in indices:
            results.append({
                'topic_id': self.bm25_data['topics'][idx],
                'topic_name': self.bm25_data['topic_names'][idx],
                'chunk': self.bm25_data['chunks'][idx],
                'score': float(scores[idx])
            })
        return results

# -----------------------------------------------------------------------
# EVALUATION ENGINE
# -----------------------------------------------------------------------

def evaluate_search_strategy(searcher: HybridSearcher, strategy: str, statements: List[Tuple[str, int]], 
                           config: OptimizationConfig) -> List[Dict]:
    """Evaluate a specific search strategy"""
    results = []
    
    for statement, true_topic in tqdm(statements, desc=f"Evaluating {strategy}"):
        # Perform search based on strategy
        if strategy == "bm25_only":
            search_results = searcher.search_bm25_only(statement, config.top_k)
        elif strategy == "semantic_only":
            search_results = searcher.search_semantic_only(statement, config.top_k)
        elif strategy.startswith("linear_"):
            weight = float(strategy.split("_")[1])
            search_results = searcher.search_linear_fusion(statement, weight, config.top_k)
        elif strategy == "rrf":
            search_results = searcher.search_rrf(statement, config.top_k)
        elif strategy == "adaptive":
            search_results = searcher.search_adaptive(statement, config.top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Deduplicate by topic_id (keep highest score)
        topic_results = {}
        for result in search_results:
            topic_id = result['topic_id']
            if topic_id not in topic_results or result['score'] > topic_results[topic_id]['score']:
                topic_results[topic_id] = result
        
        # Sort by score
        sorted_results = sorted(topic_results.values(), key=lambda x: x['score'], reverse=True)
        
        # Find rank of correct topic
        rank_correct = 0
        for rank, result in enumerate(sorted_results, 1):
            if result['topic_id'] == true_topic:
                rank_correct = rank
                break
        
        # Extract top scores for analysis
        top_scores = [r['score'] for r in sorted_results[:5]]
        
        results.append({
            'statement': statement,
            'true_topic': true_topic,
            'predicted_topic': sorted_results[0]['topic_id'] if sorted_results else -1,
            'rank_correct': rank_correct,
            'top_results': sorted_results[:3],
            'top_scores': top_scores
        })
    
    return results

# -----------------------------------------------------------------------
# ADAPTIVE EXPLORATION FUNCTIONS
# -----------------------------------------------------------------------

def detect_breakthrough(config: Dict, threshold: float) -> bool:
    """Detect if a configuration represents a breakthrough worth exploring"""
    return config.get('top1_accuracy', 0.0) >= threshold

def generate_exploration_configs(breakthrough_config: Dict, opt_config: OptimizationConfig) -> List[Dict]:
    """Generate exploration configurations around a breakthrough"""
    explorations = []
    
    chunk_size = breakthrough_config['chunk_size']
    overlap = breakthrough_config['overlap']
    model = breakthrough_config['model']
    strategy = breakthrough_config['strategy']
    
    print(f"\n🔍 BREAKTHROUGH DETECTED! Exploring around:")
    print(f"   🎯 Config: {model.split('/')[-1]} + {strategy} (cs={chunk_size}, ov={overlap})")
    print(f"   📊 Accuracy: {breakthrough_config['top1_accuracy']:.3f}")
    
    # 1. BM25 PARAMETER EXPLORATION around successful config
    chunk_variations = []
    overlap_variations = []
    radius = opt_config.exploration_radius
    
    # Chunk size variations (±radius)
    for delta in range(-radius, radius + 1):
        new_chunk = chunk_size + delta * 16  # 16-unit steps
        if 64 <= new_chunk <= 192:  # Valid range
            chunk_variations.append(new_chunk)
    
    # Overlap variations (±radius around current ratio)
    current_ratio = overlap / chunk_size
    for delta in range(-radius, radius + 1):
        new_ratio = current_ratio + delta * 0.05  # 5% steps
        if 0.0 <= new_ratio <= 0.3:  # Valid range
            overlap_variations.append(new_ratio)
    
    # Generate BM25 exploration configs
    for new_chunk in chunk_variations:
        for new_ratio in overlap_variations:
            new_overlap = int(new_chunk * new_ratio)
            explorations.append({
                'type': 'bm25_exploration',
                'chunk_size': new_chunk,
                'overlap': new_overlap,
                'overlap_ratio': new_ratio,
                'model': model,
                'strategy': strategy,
                'source': breakthrough_config['config_key']
            })
    
    # 2. FUSION STRATEGY EXPLORATION if linear strategy worked
    if strategy.startswith('linear_'):
        current_weight = float(strategy.split('_')[1])
        print(f"   🔬 Exploring linear weights around {current_weight}")
        
        # Test finer-grained weights around successful one
        weight_variations = []
        for delta in [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]:
            new_weight = current_weight + delta
            if 0.1 <= new_weight <= 0.9:  # Valid range
                weight_variations.append(new_weight)
        
        for new_weight in weight_variations:
            explorations.append({
                'type': 'strategy_exploration',
                'chunk_size': chunk_size,
                'overlap': overlap,
                'overlap_ratio': overlap / chunk_size,
                'model': model,
                'strategy': f'linear_{new_weight:.2f}',
                'source': breakthrough_config['config_key']
            })
    
    # 3. MODEL ARCHITECTURE EXPLORATION
    if 'distilroberta' in model.lower():
        # If DistilRoBERTa worked well, try related models
        related_models = [
            "sentence-transformers/all-roberta-large-v1",
            "sentence-transformers/paraphrase-distilroberta-base-v2",
        ]
        for related_model in related_models:
            explorations.append({
                'type': 'model_exploration', 
                'chunk_size': chunk_size,
                'overlap': overlap,
                'overlap_ratio': overlap / chunk_size,
                'model': related_model,
                'strategy': strategy,
                'source': breakthrough_config['config_key']
            })
    
    # Limit exploration size
    if len(explorations) > opt_config.max_exploration_configs:
        explorations = explorations[:opt_config.max_exploration_configs]
        print(f"   ⚡ Limited exploration to {len(explorations)} configs")
    
    print(f"   🚀 Generated {len(explorations)} exploration configs")
    return explorations

def execute_exploration(exploration_configs: List[Dict], statements: List, opt_config: OptimizationConfig,
                       all_results: Dict, best_configs: List) -> List[Dict]:
    """Execute exploration configurations and return new breakthroughs"""
    new_breakthroughs = []
    
    print(f"\n" + "🔬" * 60)
    print(f"🔬 ADAPTIVE EXPLORATION PHASE ({len(exploration_configs)} configs)")
    print(f"🔬" * 60)
    
    for exp_idx, exp_config in enumerate(exploration_configs, 1):
        print(f"\n🔍 EXPLORATION {exp_idx}/{len(exploration_configs)}: {exp_config['type']}")
        print(f"   📊 Testing: {exp_config['model'].split('/')[-1]} + {exp_config['strategy']}")
        print(f"   ⚙️  BM25: chunk_size={exp_config['chunk_size']}, overlap={exp_config['overlap']}")
        print(f"   🎯 Source: {exp_config['source']}")
        
        # Build/load BM25 and semantic indices
        bm25_data = build_bm25_index(exp_config['chunk_size'], exp_config['overlap'], opt_config.use_condensed_topics)
        semantic_data = build_semantic_index(exp_config['model'], bm25_data, opt_config.cache_embeddings)
        
        if semantic_data is None:
            print(f"   ❌ Skipping - could not load {exp_config['model']}")
            continue
        
        # Create searcher and evaluate
        searcher = HybridSearcher(bm25_data, semantic_data)
        start_time = time.time()
        eval_results = evaluate_search_strategy(searcher, exp_config['strategy'], statements, opt_config)
        elapsed = time.time() - start_time
        
        # Calculate metrics
        metrics = calculate_metrics(eval_results)
        metrics['evaluation_time'] = elapsed
        metrics['time_per_query'] = elapsed / len(statements)
        
        # Store results
        config_key = f"EXPLORATION_{exp_config['model'].split('/')[-1]}_{exp_config['strategy']}_cs{exp_config['chunk_size']}_ov{exp_config['overlap']}"
        all_results[config_key] = {
            'metrics': metrics,
            'config': {
                'model': exp_config['model'],
                'strategy': exp_config['strategy'],
                'chunk_size': exp_config['chunk_size'],
                'overlap': exp_config['overlap'],
                'overlap_ratio': exp_config['overlap_ratio'],
                'use_condensed_topics': opt_config.use_condensed_topics,
                'exploration_type': exp_config['type'],
                'source_breakthrough': exp_config['source']
            },
            'detailed_results': eval_results if opt_config.save_detailed_results else None
        }
        
        # Add to best configs
        exploration_result = {
            'config_key': config_key,
            'model': exp_config['model'],
            'strategy': exp_config['strategy'],
            'chunk_size': exp_config['chunk_size'],
            'overlap': exp_config['overlap'],
            'overlap_ratio': exp_config['overlap_ratio'],
            'top1_accuracy': metrics['top1_accuracy'],
            'top2_accuracy': metrics['top2_accuracy'],
            'top3_accuracy': metrics['top3_accuracy'],
            'top4_accuracy': metrics['top4_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'mrr': metrics['mrr'],
            'avg_separation': metrics['avg_separation_when_correct'],
            'score_gap_p90': metrics['score_gaps']['percentiles']['p90'],
            'time_per_query': metrics['time_per_query'],
            'exploration_type': exp_config['type'],
            'source_breakthrough': exp_config['source']
        }
        best_configs.append(exploration_result)
        
        # Check if this exploration is also a breakthrough
        if detect_breakthrough(exploration_result, opt_config.breakthrough_threshold):
            new_breakthroughs.append(exploration_result)
            print(f"   🚀 NEW BREAKTHROUGH: T1={metrics['top1_accuracy']:.3f}!")
        
        print(f"   ✅ {exp_config['strategy']}: T1: {metrics['top1_accuracy']:.3f}, "
              f"T2: {metrics['top2_accuracy']:.3f}, "
              f"T3: {metrics['top3_accuracy']:.3f}, "
              f"T5: {metrics['top5_accuracy']:.3f}, "
              f"MRR: {metrics['mrr']:.3f} | {elapsed:.1f}s")
    
    return new_breakthroughs

# -----------------------------------------------------------------------
# INCREMENTAL SAVING FUNCTIONS
# -----------------------------------------------------------------------

def save_partial_results(all_results: Dict, best_configs: List, opt_config: OptimizationConfig, 
                        phase: str, additional_info: Dict = None) -> str:
    """Save partial results during optimization"""
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = f"optimization_results_partial_{phase}_{timestamp}.json"
    
    # Basic results structure
    partial_summary = {
        'phase': phase,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'embedding_models': opt_config.embedding_models,
            'fusion_strategies': opt_config.fusion_strategies,
            'chunk_sizes': opt_config.chunk_sizes,
            'overlap_ratios': opt_config.overlap_ratios,
            'use_condensed_topics': opt_config.use_condensed_topics,
            'optimize_bm25': opt_config.optimize_bm25,
            'fast_mode': opt_config.fast_mode,
            'top_bm25_configs': opt_config.top_bm25_configs,
        },
        'progress': {
            'total_configs_tested': len(best_configs),
            'best_configurations': best_configs,
        },
        'detailed_results': all_results
    }
    
    # Add phase-specific info
    if additional_info:
        partial_summary.update(additional_info)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(partial_summary, f, indent=2, default=str)
    
    return output_file

def print_cancel_instructions(phase: str, saved_file: str):
    """Print instructions for accessing results if canceled"""
    print(f"\n💾 PROGRESS SAVED: {saved_file}")
    print(f"🛡️  SAFE TO CANCEL: Press Ctrl+C anytime to stop")
    print(f"📂 Results so far: python -c \"import json; print(json.load(open('{saved_file}'))['progress']['total_configs_tested'], 'configs tested')\"")

# -----------------------------------------------------------------------
# MAIN OPTIMIZATION PIPELINE
# -----------------------------------------------------------------------

def run_optimization():
    """Run comprehensive optimization of topic model"""
    start_time = time.time()
    print("🚀 Starting Topic Model Optimization")
    print("=" * 60)
    
    # Configuration - modify these settings as needed
    opt_config = OptimizationConfig(
        optimize_bm25=True,  # Find best BM25 parameters first, then test fusion strategies
        use_condensed_topics=True,  # True for condensed, False for original topics
        max_samples=200,  # Number of statements to evaluate (max 200)
        cache_embeddings=True,  # Cache embeddings for faster re-runs
        fast_mode=False,  # Enable full exploration to find breakthrough configs like 91%
        top_bm25_configs=10,  # Test semantic fusion on top 10 BM25 configs for thorough search
        sample_strategies=False,  # Use all strategies to find the best combinations
        target_runtime_hours=8.0,  # Allow longer runtime for breakthrough discovery
        enable_adaptive_exploration=True,  # Detect and explore around promising results
        breakthrough_threshold=0.905,  # Look for configs above 90.5% (like your 91% finding)
        exploration_radius=2,  # Explore ±2 parameter variations around breakthroughs
        max_exploration_configs=50  # Allow substantial exploration around each breakthrough
    )
    
    print(f"📋 Configuration:")
    print(f"   🗂️ Topics: {'Condensed' if opt_config.use_condensed_topics else 'Original'}")
    print(f"   🔧 BM25 Optimization: {'Enabled' if opt_config.optimize_bm25 else 'Disabled (using best known config)'}")
    print(f"   📊 Samples: {opt_config.max_samples}")
    print(f"   🧠 Models: {len(opt_config.embedding_models)}")
    print(f"   🔍 Strategies: {len(opt_config.fusion_strategies)}")
    print(f"   ⚡ Mode: {'Fast hierarchical' if opt_config.fast_mode else 'Full exploration'}")
    print(f"   🔍 Adaptive exploration: {'Enabled' if opt_config.enable_adaptive_exploration else 'Disabled'}")
    if opt_config.enable_adaptive_exploration:
        print(f"      🎯 Breakthrough threshold: {opt_config.breakthrough_threshold:.3f}")
        print(f"      🔬 Exploration radius: ±{opt_config.exploration_radius}")
        print(f"      📊 Max exploration configs: {opt_config.max_exploration_configs}")
    print(f"   💾 Auto-save: Enabled (progress saved continuously)")
    print(f"   🛡️  Safe to cancel: Press Ctrl+C anytime - progress is preserved!")
    
    bm25_configs_count = len(opt_config.chunk_sizes) * len(opt_config.overlap_ratios)
    semantic_strategies = [s for s in opt_config.fusion_strategies if s != "bm25_only"]
    
    if opt_config.fast_mode:
        # Fast mode: only test semantic fusion on top N BM25 configs
        selected_bm25_count = min(opt_config.top_bm25_configs, bm25_configs_count)
        semantic_configs = selected_bm25_count * len(opt_config.embedding_models) * len(semantic_strategies)
        total_configs = bm25_configs_count + semantic_configs  # All BM25 + selected semantic
        
        # BM25 is fast (~10s per config), semantic fusion is slow (~2min per config)
        bm25_time = bm25_configs_count * 0.2  # 10-15s per BM25-only config
        semantic_time = semantic_configs * 1.5   # 1-2min per semantic config
        estimated_total = bm25_time + semantic_time
        
        print(f"   🚀 FAST MODE ENABLED:")
        print(f"   ⏱️ BM25-only Configs: {bm25_configs_count} (~{bm25_time:.0f} minutes)")
        print(f"   ⏱️ Semantic Configs: {semantic_configs} (~{semantic_time//60:.0f}-{semantic_time//60*1.5:.0f} hours)")
        print(f"   ⏱️ Total Configs: {total_configs} (~{estimated_total//60:.0f}-{estimated_total//60*1.5:.0f} hours)")
        print(f"   ⚡ Semantic testing limited to TOP {selected_bm25_count} BM25 configs")
        print(f"   ⚡ Target runtime: {opt_config.target_runtime_hours:.1f} hours")
    else:
        # Full mode: test all combinations
        semantic_configs = bm25_configs_count * len(opt_config.embedding_models) * len(semantic_strategies)
        total_configs = bm25_configs_count + semantic_configs
        
        bm25_time = bm25_configs_count * 0.2
        semantic_time = semantic_configs * 1.5
        estimated_total = bm25_time + semantic_time
        
        print(f"   🔧 FULL MODE:")
        print(f"   ⏱️ BM25-only Configs: {bm25_configs_count} (~{bm25_time:.0f} minutes)")
        print(f"   ⏱️ Semantic Configs: {semantic_configs} (~{semantic_time//60:.0f}-{semantic_time//60*1.5:.0f} hours)")
        print(f"   ⏱️ Total Configs: {total_configs} (~{estimated_total//60:.0f}-{estimated_total//60*1.5:.0f} hours)")
    
    if opt_config.optimize_bm25:
        print(f"   🚀 Deep BM25 search: {len(opt_config.chunk_sizes)} chunk sizes × {len(opt_config.overlap_ratios)} overlaps")
    
    # Load test data
    print("📚 Loading evaluation data...")
    statements = load_statements()
    if opt_config.max_samples and opt_config.max_samples < len(statements):
        statements = statements[:opt_config.max_samples]
    print(f"📊 Evaluating on {len(statements)} statements")
    
    # Initial save
    initial_info = {
        'startup_info': {
            'total_statements': len(statements),
            'estimated_runtime_hours': opt_config.target_runtime_hours if opt_config.fast_mode else 'unknown',
            'fast_mode': opt_config.fast_mode
        }
    }
    initial_file = save_partial_results({}, [], opt_config, "startup", initial_info)
    print(f"💾 Initial save: {initial_file}")
    print(f"🛡️  Safe to proceed - progress will be automatically saved!")
    
    # Store all results
    all_results = {}
    best_configs = []
    
    # Generate BM25 configurations to test
    bm25_configs = []
    for chunk_size in opt_config.chunk_sizes:
        for overlap_ratio in opt_config.overlap_ratios:
            overlap = int(chunk_size * overlap_ratio)
            bm25_configs.append((chunk_size, overlap))
    
    if opt_config.optimize_bm25:
        print(f"\n🔧 Step 1: BM25 Optimization - Testing {len(bm25_configs)} configurations:")
        for chunk_size, overlap in bm25_configs[:3]:  # Show first 3
            print(f"   • chunk_size={chunk_size}, overlap={overlap} ({overlap/chunk_size:.1%})")
        if len(bm25_configs) > 3:
            print(f"   • ... and {len(bm25_configs)-3} more")
        print(f"🔧 Step 2: Then test semantic fusion on best BM25 configs")
    else:
        print(f"\n🔧 Using known best BM25 config: chunk_size=128, overlap=12")
    
    # PHASE 1: EVALUATE ALL BM25 CONFIGURATIONS
    print(f"\n" + "=" * 80)
    print(f"🔥 PHASE 1: BM25 OPTIMIZATION ({len(bm25_configs)} configurations)")
    print(f"=" * 80)
    
    bm25_results = []  # Store BM25 results for ranking
    
    for bm25_idx, (chunk_size, overlap) in enumerate(bm25_configs):
        print(f"\n📊 BM25 CONFIG {bm25_idx+1}/{len(bm25_configs)} | chunk_size={chunk_size}, overlap={overlap} | Progress: {(bm25_idx+1)/len(bm25_configs)*100:.1f}%")
        print(f"{'='*60}")
        
        # Build BM25 index for this configuration
        bm25_data = build_bm25_index(chunk_size, overlap, opt_config.use_condensed_topics)
        print(f"   ✅ BM25 index ready with {len(bm25_data['chunks'])} chunks")
        
        # EVALUATE BM25-ONLY ONCE (no redundancy across embedding models)
        print(f"\n🔥 BM25-ONLY EVALUATION (independent of embedding models)")
        bm25_searcher = HybridSearcher(bm25_data, None)
        start_time_eval = time.time()
        eval_results = evaluate_search_strategy(bm25_searcher, "bm25_only", statements, opt_config)
        elapsed = time.time() - start_time_eval
        
        # Calculate metrics for BM25-only
        metrics = calculate_metrics(eval_results)
        metrics['evaluation_time'] = elapsed
        metrics['time_per_query'] = elapsed / len(statements)
        
        # Store BM25-only results
        config_key = f"BM25_ONLY_bm25_only_cs{chunk_size}_ov{overlap}"
        all_results[config_key] = {
            'metrics': metrics,
            'config': {
                'model': 'BM25_ONLY',
                'strategy': 'bm25_only',
                'chunk_size': chunk_size,
                'overlap': overlap,
                'overlap_ratio': overlap / chunk_size,
                'use_condensed_topics': opt_config.use_condensed_topics
            },
            'detailed_results': eval_results if opt_config.save_detailed_results else None
        }
        
        # Add to best configs
        bm25_config_result = {
            'config_key': config_key,
            'model': 'BM25_ONLY',
            'strategy': 'bm25_only',
            'chunk_size': chunk_size,
            'overlap': overlap,
            'overlap_ratio': overlap / chunk_size,
            'top1_accuracy': metrics['top1_accuracy'],
            'top2_accuracy': metrics['top2_accuracy'],
            'top3_accuracy': metrics['top3_accuracy'],
            'top4_accuracy': metrics['top4_accuracy'],
            'top5_accuracy': metrics['top5_accuracy'],
            'mrr': metrics['mrr'],
            'avg_separation': metrics['avg_separation_when_correct'],
            'score_gap_p90': metrics['score_gaps']['percentiles']['p90'],
            'time_per_query': metrics['time_per_query'],
            'bm25_data': bm25_data  # Store for later use
        }
        best_configs.append(bm25_config_result)
        bm25_results.append(bm25_config_result)
        
        print(f"   ✅ BM25-only: T1: {metrics['top1_accuracy']:.3f}, "
              f"T2: {metrics['top2_accuracy']:.3f}, "
              f"T3: {metrics['top3_accuracy']:.3f}, "
              f"T5: {metrics['top5_accuracy']:.3f}, "
              f"MRR: {metrics['mrr']:.3f} | {elapsed:.1f}s")
    
    # PHASE 1 COMPLETE - ANALYZE RESULTS
    print(f"\n🎉 PHASE 1 COMPLETE! Evaluated {len(bm25_configs)} BM25 configurations")
    
    # Sort BM25 results by top-1 accuracy
    bm25_results.sort(key=lambda x: x['top1_accuracy'], reverse=True)
    
    print(f"\n🏆 TOP BM25 CONFIGURATIONS:")
    print("   Rank | CS  | OV  | T1     | T2     | T3     | MRR")
    print("   " + "-" * 45)
    for i, result in enumerate(bm25_results[:5], 1):
        print(f"   {i:<4} | {result['chunk_size']:<3} | {result['overlap']:<3} | "
              f"{result['top1_accuracy']:.3f}  | {result['top2_accuracy']:.3f}  | "
              f"{result['top3_accuracy']:.3f}  | {result['mrr']:.3f}")
    
    # SAVE PHASE 1 RESULTS
    phase1_info = {
        'phase1_results': {
            'bm25_rankings': [{'chunk_size': r['chunk_size'], 'overlap': r['overlap'], 
                             'top1_accuracy': r['top1_accuracy']} for r in bm25_results[:10]],
            'best_bm25_accuracy': bm25_results[0]['top1_accuracy'],
            'total_bm25_configs': len(bm25_configs)
        }
    }
    phase1_file = save_partial_results(all_results, best_configs, opt_config, "phase1_bm25_complete", phase1_info)
    print_cancel_instructions("Phase 1", phase1_file)
    
    # SMART SELECTION: Choose top BM25 configs for semantic fusion
    if opt_config.fast_mode:
        selected_bm25 = bm25_results[:opt_config.top_bm25_configs]
        print(f"\n🚀 FAST MODE: Testing semantic fusion on TOP {len(selected_bm25)} BM25 configs only")
        print(f"   ⚡ Skipping {len(bm25_results) - len(selected_bm25)} lower-performing BM25 configs")
    else:
        selected_bm25 = bm25_results
        print(f"\n🔧 FULL MODE: Testing semantic fusion on all {len(selected_bm25)} BM25 configs")
    
    # PHASE 2: SEMANTIC FUSION ON SELECTED BM25 CONFIGS
    print(f"\n" + "=" * 80)
    print(f"🧠 PHASE 2: SEMANTIC FUSION ({len(selected_bm25)} BM25 configs × {len(opt_config.embedding_models)} models × {len([s for s in opt_config.fusion_strategies if s != 'bm25_only'])} strategies)")
    print(f"=" * 80)
    
    semantic_strategies = [s for s in opt_config.fusion_strategies if s != "bm25_only"]
    total_semantic_configs = len(selected_bm25) * len(opt_config.embedding_models) * len(semantic_strategies)
    semantic_config_idx = 0
    
    for bm25_rank, bm25_result in enumerate(selected_bm25, 1):
        chunk_size = bm25_result['chunk_size']
        overlap = bm25_result['overlap']
        bm25_data = bm25_result['bm25_data']
        
        print(f"\n📊 BM25 CONFIG {bm25_rank}/{len(selected_bm25)} (RANK #{bm25_rank} overall) | chunk_size={chunk_size}, overlap={overlap}")
        print(f"   🏆 BM25-only performance: T1={bm25_result['top1_accuracy']:.3f}")
        print(f"{'='*70}")
        
        # Test each embedding model with this BM25 config
        for model_idx, model_name in enumerate(opt_config.embedding_models):
            print(f"\n🤖 MODEL {model_idx+1}/{len(opt_config.embedding_models)}: {model_name.split('/')[-1]}")
            print(f"   BM25 Config: {bm25_rank}/{len(selected_bm25)} | Model: {model_idx+1}/{len(opt_config.embedding_models)}")
            
            # Build semantic index
            semantic_data = build_semantic_index(model_name, bm25_data, opt_config.cache_embeddings)
            if semantic_data is None:
                print(f"   ❌ Skipping {model_name} due to loading error")
                continue
            
            # Create searcher
            searcher = HybridSearcher(bm25_data, semantic_data)
            
            # Test each fusion strategy (skip BM25-only since already evaluated)
            total_strategies = len(semantic_strategies)
            
            for strategy_idx, strategy in enumerate(semantic_strategies):
                semantic_config_idx += 1
                
                # Calculate overall progress
                bm25_only_configs = len(bm25_configs)  # Total BM25-only configs evaluated
                total_all_configs = bm25_only_configs + total_semantic_configs
                configs_completed = bm25_only_configs + semantic_config_idx
                
                print(f"\n🔍 STRATEGY {strategy_idx+1}/{total_strategies}: {strategy}")
                print(f"   📊 OVERALL PROGRESS: {configs_completed}/{total_all_configs} ({configs_completed/total_all_configs*100:.1f}%)")
                print(f"   📍 BM25: {bm25_rank}/{len(selected_bm25)} | Model: {model_idx+1}/{len(opt_config.embedding_models)} | Strategy: {strategy_idx+1}/{total_strategies}")
                
                config_key = f"{model_name}_{strategy}_cs{chunk_size}_ov{overlap}"
                
                # Skip semantic strategies if no semantic model
                if strategy == "semantic_only" and semantic_data is None:
                    print(f"   ⏭️  Skipping (no semantic model)")
                    semantic_config_idx -= 1  # Don't count skipped configs
                    continue
                
                # Run evaluation
                start_time_eval = time.time()
                eval_results = evaluate_search_strategy(searcher, strategy, statements, opt_config)
                elapsed = time.time() - start_time_eval
                
                # Calculate metrics
                metrics = calculate_metrics(eval_results)
                metrics['evaluation_time'] = elapsed
                metrics['time_per_query'] = elapsed / len(statements)
                
                # Store results
                all_results[config_key] = {
                    'metrics': metrics,
                    'config': {
                        'model': model_name,
                        'strategy': strategy,
                        'chunk_size': chunk_size,
                        'overlap': overlap,
                        'overlap_ratio': overlap / chunk_size,
                        'use_condensed_topics': opt_config.use_condensed_topics
                    },
                    'detailed_results': eval_results if opt_config.save_detailed_results else None
                }
                
                # Add to best configs list
                best_configs.append({
                    'config_key': config_key,
                    'model': model_name,
                    'strategy': strategy,
                    'chunk_size': chunk_size,
                    'overlap': overlap,
                    'overlap_ratio': overlap / chunk_size,
                    'top1_accuracy': metrics['top1_accuracy'],
                    'top2_accuracy': metrics['top2_accuracy'],
                    'top3_accuracy': metrics['top3_accuracy'],
                    'top4_accuracy': metrics['top4_accuracy'],
                    'top5_accuracy': metrics['top5_accuracy'],
                    'mrr': metrics['mrr'],
                    'avg_separation': metrics['avg_separation_when_correct'],
                    'score_gap_p90': metrics['score_gaps']['percentiles']['p90'],
                    'time_per_query': metrics['time_per_query']
                })
                
                # Print summary
                print(f"   ✅ {strategy}: T1: {metrics['top1_accuracy']:.3f}, "
                      f"T2: {metrics['top2_accuracy']:.3f}, "
                      f"T3: {metrics['top3_accuracy']:.3f}, "
                      f"T5: {metrics['top5_accuracy']:.3f}, "
                      f"MRR: {metrics['mrr']:.3f} | {elapsed:.1f}s")
        
        # Phase completion markers  
        print(f"\n✅ COMPLETED BM25 CONFIG {bm25_rank}/{len(selected_bm25)}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # SAVE PROGRESS AFTER EACH BM25 CONFIG IN PHASE 2
        if bm25_rank % 1 == 0:  # Save after each BM25 config (change to 2 or 3 for less frequent saves)
            phase2_info = {
                'phase2_progress': {
                    'completed_bm25_configs': bm25_rank,
                    'total_bm25_configs': len(selected_bm25),
                    'completed_semantic_configs': semantic_config_idx,
                    'current_best_accuracy': max([c['top1_accuracy'] for c in best_configs]) if best_configs else 0.0
                }
            }
            phase2_file = save_partial_results(all_results, best_configs, opt_config, f"phase2_progress_bm25_{bm25_rank}", phase2_info)
            print(f"💾 Progress saved: {phase2_file}")
        
        # Clean up BM25 data to save memory (keep it only in selected configs)
        bm25_result['bm25_data'] = None  # Remove after use
    
    # ADAPTIVE EXPLORATION PHASE
    if opt_config.enable_adaptive_exploration:
        print(f"\n" + "🔍" * 80)
        print(f"🔍 CHECKING FOR BREAKTHROUGHS (threshold: {opt_config.breakthrough_threshold:.3f})")
        print(f"🔍" * 80)
        
        # Find all breakthroughs in current results
        breakthroughs = []
        for config in best_configs:
            if detect_breakthrough(config, opt_config.breakthrough_threshold):
                breakthroughs.append(config)
        
        if breakthroughs:
            print(f"\n🚀 FOUND {len(breakthroughs)} BREAKTHROUGH(S)!")
            for i, breakthrough in enumerate(breakthroughs, 1):
                print(f"   #{i}: {breakthrough['model'].split('/')[-1]} + {breakthrough['strategy']} → T1: {breakthrough['top1_accuracy']:.3f}")
            
            # Generate exploration configs for all breakthroughs
            all_explorations = []
            for breakthrough in breakthroughs:
                exploration_configs = generate_exploration_configs(breakthrough, opt_config)
                all_explorations.extend(exploration_configs)
            
            if all_explorations:
                # Execute exploration phase
                print(f"\n🔬 EXECUTING ADAPTIVE EXPLORATION...")
                exploration_breakthroughs = execute_exploration(all_explorations, statements, opt_config, all_results, best_configs)
                
                # Save exploration results
                exploration_info = {
                    'exploration_results': {
                        'original_breakthroughs': len(breakthroughs),
                        'exploration_configs_tested': len(all_explorations),
                        'new_breakthroughs_found': len(exploration_breakthroughs),
                        'best_exploration_accuracy': max([c['top1_accuracy'] for c in exploration_breakthroughs]) if exploration_breakthroughs else None
                    }
                }
                exploration_file = save_partial_results(all_results, best_configs, opt_config, "exploration_complete", exploration_info)
                print(f"💾 Exploration results saved: {exploration_file}")
                
                # RECURSIVE EXPLORATION: If new breakthroughs found, potentially explore further
                if exploration_breakthroughs and len(exploration_breakthroughs) > 0:
                    improved_threshold = opt_config.breakthrough_threshold + 0.01  # Raise bar slightly
                    print(f"\n🚀 NEW BREAKTHROUGHS FOUND! Checking for super-breakthroughs (threshold: {improved_threshold:.3f})")
                    
                    super_breakthroughs = [b for b in exploration_breakthroughs if b['top1_accuracy'] >= improved_threshold]
                    if super_breakthroughs and len(super_breakthroughs) <= 2:  # Limit recursive exploration
                        print(f"🔥 SUPER-BREAKTHROUGH DETECTED! Running focused exploration...")
                        focused_explorations = []
                        for super_b in super_breakthroughs:
                            # More focused exploration around super-breakthroughs
                            temp_config = OptimizationConfig(**opt_config.__dict__)
                            temp_config.exploration_radius = 1  # Smaller radius
                            temp_config.max_exploration_configs = 20  # Fewer configs
                            focused_configs = generate_exploration_configs(super_b, temp_config)
                            focused_explorations.extend(focused_configs)
                        
                        if focused_explorations:
                            print(f"🎯 FOCUSED EXPLORATION: {len(focused_explorations)} configs")
                            execute_exploration(focused_explorations, statements, opt_config, all_results, best_configs)
        else:
            print(f"\n📊 No breakthroughs found above {opt_config.breakthrough_threshold:.3f} threshold")
            print(f"   🎯 Current best: {max([c['top1_accuracy'] for c in best_configs]):.3f}")
            print(f"   💡 Consider lowering breakthrough_threshold or running longer")
    
    # Completion marker
    print(f"\n" + "=" * 80)
    print(f"🎉 OPTIMIZATION COMPLETE!")
    print(f"   📊 Total configurations tested: {len(best_configs)}")
    print(f"   📊 BM25-only configs: {bm25_configs_count}")
    print(f"   📊 Semantic fusion configs: {len(best_configs) - bm25_configs_count}")
    print(f"   ⏱️  Total runtime: {(time.time() - start_time) / 3600:.1f} hours")
    print(f"=" * 80)
    
    # Clean up any remaining bm25_data references before results
    for config in best_configs:
        if 'bm25_data' in config:
            del config['bm25_data']
    
    # Find and display best configurations
    print(f"\n🏆 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    # Sort by top-1 accuracy first, then MRR
    best_configs.sort(key=lambda x: (x['top1_accuracy'], x['mrr']), reverse=True)
    
    print("\n🥇 TOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<4} {'Model':<25} {'Strategy':<12} {'CS':<3} {'OV':<3} {'T1':<6} {'T2':<6} {'T3':<6} {'T4':<6} {'T5':<6} {'MRR':<6}")
    print("-" * 95)
    
    for i, config in enumerate(best_configs[:10], 1):
        model_short = config['model'].split('/')[-1][:20]
        print(f"{i:<4} {model_short:<25} {config['strategy']:<12} "
              f"{config['chunk_size']:<3} {config['overlap']:<3} "
              f"{config['top1_accuracy']:.3f}  {config['top2_accuracy']:.3f}  "
              f"{config['top3_accuracy']:.3f}  {config['top4_accuracy']:.3f}  "
              f"{config['top5_accuracy']:.3f}  {config['mrr']:.3f}")
    
    # COMPREHENSIVE TREND ANALYSIS
    print("\n" + "=" * 80)
    print("📊 TREND ANALYSIS & OPTIMIZATION INSIGHTS")
    print("=" * 80)
    
    # 1. EMBEDDING MODEL PERFORMANCE ANALYSIS
    print("\n🤖 EMBEDDING MODEL PERFORMANCE:")
    model_performance = {}
    for config in best_configs:
        model = config['model']
        if model != 'BM25_ONLY':  # Skip BM25-only for model comparison
            if model not in model_performance:
                model_performance[model] = []
            model_performance[model].append(config['top1_accuracy'])
    
    model_stats = []
    for model, accuracies in model_performance.items():
        model_stats.append({
            'model': model,
            'avg_accuracy': np.mean(accuracies),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies),
            'std_accuracy': np.std(accuracies),
            'count': len(accuracies)
        })
    
    model_stats.sort(key=lambda x: x['avg_accuracy'], reverse=True)
    print("   Model                     | Avg T1  | Max T1  | Min T1  | Std    | Configs")
    print("   " + "-" * 65)
    for stat in model_stats:
        model_name = stat['model'].split('/')[-1][:25]
        print(f"   {model_name:<25} | {stat['avg_accuracy']:.3f}   | {stat['max_accuracy']:.3f}   | "
              f"{stat['min_accuracy']:.3f}   | {stat['std_accuracy']:.3f}  | {stat['count']}")
    
    # 2. FUSION STRATEGY PERFORMANCE ANALYSIS
    print("\n🔍 FUSION STRATEGY PERFORMANCE:")
    strategy_performance = {}
    for config in best_configs:
        strategy = config['strategy']
        if strategy not in strategy_performance:
            strategy_performance[strategy] = []
        strategy_performance[strategy].append(config['top1_accuracy'])
    
    strategy_stats = []
    for strategy, accuracies in strategy_performance.items():
        strategy_stats.append({
            'strategy': strategy,
            'avg_accuracy': np.mean(accuracies),
            'max_accuracy': np.max(accuracies),
            'min_accuracy': np.min(accuracies),
            'std_accuracy': np.std(accuracies),
            'count': len(accuracies)
        })
    
    strategy_stats.sort(key=lambda x: x['avg_accuracy'], reverse=True)
    print("   Strategy      | Avg T1  | Max T1  | Min T1  | Std    | Configs")
    print("   " + "-" * 55)
    for stat in strategy_stats:
        print(f"   {stat['strategy']:<12} | {stat['avg_accuracy']:.3f}   | {stat['max_accuracy']:.3f}   | "
              f"{stat['min_accuracy']:.3f}   | {stat['std_accuracy']:.3f}  | {stat['count']}")
    
    # 3. BM25 PARAMETER OPTIMIZATION ANALYSIS
    print("\n🔧 BM25 PARAMETER ANALYSIS:")
    
    # Chunk size analysis
    chunk_performance = {}
    for config in best_configs:
        chunk_size = config['chunk_size']
        if chunk_size not in chunk_performance:
            chunk_performance[chunk_size] = []
        chunk_performance[chunk_size].append(config['top1_accuracy'])
    
    print("   📏 CHUNK SIZE PERFORMANCE:")
    print("   Size | Avg T1  | Max T1  | Configs | Recommendation")
    print("   " + "-" * 50)
    chunk_stats = []
    for chunk_size, accuracies in chunk_performance.items():
        avg_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        count = len(accuracies)
        chunk_stats.append((chunk_size, avg_acc, max_acc, count))
    
    chunk_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by avg accuracy
    for i, (chunk_size, avg_acc, max_acc, count) in enumerate(chunk_stats[:5]):
        recommendation = "⭐ BEST" if i == 0 else "✅ GOOD" if i < 3 else ""
        print(f"   {chunk_size:<4} | {avg_acc:.3f}   | {max_acc:.3f}   | {count:<7} | {recommendation}")
    
    # Overlap analysis
    overlap_performance = {}
    for config in best_configs:
        overlap_ratio = round(config['overlap_ratio'], 3)
        if overlap_ratio not in overlap_performance:
            overlap_performance[overlap_ratio] = []
        overlap_performance[overlap_ratio].append(config['top1_accuracy'])
    
    print("\n   📐 OVERLAP RATIO PERFORMANCE:")
    print("   Ratio | Avg T1  | Max T1  | Configs | Recommendation")
    print("   " + "-" * 50)
    overlap_stats = []
    for overlap_ratio, accuracies in overlap_performance.items():
        avg_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        count = len(accuracies)
        overlap_stats.append((overlap_ratio, avg_acc, max_acc, count))
    
    overlap_stats.sort(key=lambda x: x[1], reverse=True)  # Sort by avg accuracy
    for i, (overlap_ratio, avg_acc, max_acc, count) in enumerate(overlap_stats[:5]):
        recommendation = "⭐ BEST" if i == 0 else "✅ GOOD" if i < 3 else ""
        print(f"   {overlap_ratio:.3f} | {avg_acc:.3f}   | {max_acc:.3f}   | {count:<7} | {recommendation}")
    
    # 4. PERFORMANCE DISTRIBUTION INSIGHTS
    print("\n📈 PERFORMANCE DISTRIBUTION:")
    all_accuracies = [config['top1_accuracy'] for config in best_configs]
    bm25_only_accuracies = [config['top1_accuracy'] for config in best_configs if config['strategy'] == 'bm25_only']
    semantic_accuracies = [config['top1_accuracy'] for config in best_configs if config['strategy'] != 'bm25_only']
    
    print(f"   📊 Overall Performance:")
    print(f"      Mean: {np.mean(all_accuracies):.3f} | Std: {np.std(all_accuracies):.3f}")
    print(f"      Best: {np.max(all_accuracies):.3f} | Worst: {np.min(all_accuracies):.3f}")
    print(f"      95th percentile: {np.percentile(all_accuracies, 95):.3f}")
    
    if bm25_only_accuracies and semantic_accuracies:
        bm25_best = np.max(bm25_only_accuracies)
        semantic_best = np.max(semantic_accuracies)
        fusion_benefit = (semantic_best - bm25_best) * 100
        print(f"\n   🚀 Fusion Benefit Analysis:")
        print(f"      Best BM25-only: {bm25_best:.3f}")
        print(f"      Best Semantic Fusion: {semantic_best:.3f}")
        print(f"      Fusion Benefit: +{fusion_benefit:.2f}% improvement")
    
    # 5. FOCUSED OPTIMIZATION RECOMMENDATIONS
    print("\n🎯 FOCUSED OPTIMIZATION RECOMMENDATIONS:")
    print("-" * 60)
    
    # Best model
    if model_stats:
        best_model = model_stats[0]
        print(f"🥇 Best Embedding Model: {best_model['model'].split('/')[-1]}")
        print(f"   📊 Avg accuracy: {best_model['avg_accuracy']:.3f} ({best_model['count']} configs)")
        
        # Best strategy for best model
        best_model_configs = [c for c in best_configs if c['model'] == best_model['model']]
        if best_model_configs:
            best_strategy_for_model = best_model_configs[0]['strategy']
            print(f"   🔍 Best strategy: {best_strategy_for_model}")
    
    # Best strategy overall
    if strategy_stats:
        best_strategy = strategy_stats[0]
        print(f"\n🔍 Best Fusion Strategy: {best_strategy['strategy']}")
        print(f"   📊 Avg accuracy: {best_strategy['avg_accuracy']:.3f} ({best_strategy['count']} configs)")
    
    # Optimal BM25 parameters
    if chunk_stats and overlap_stats:
        best_chunk = chunk_stats[0][0]
        best_overlap = overlap_stats[0][0]
        print(f"\n🔧 Optimal BM25 Parameters:")
        print(f"   📏 Chunk size: {best_chunk}")
        print(f"   📐 Overlap ratio: {best_overlap:.3f}")
    
    # Future optimization focus
    print(f"\n🚀 Next Optimization Focus:")
    if len(model_stats) > 1:
        model_diff = model_stats[0]['avg_accuracy'] - model_stats[1]['avg_accuracy']
        if model_diff < 0.01:  # Models are close
            print(f"   🤖 Models perform similarly (±{model_diff:.3f}) - focus on fusion strategies")
        else:
            print(f"   🤖 Clear model winner ({model_diff:.3f} advantage) - focus on this model")
    
    if len(strategy_stats) > 1:
        strategy_diff = strategy_stats[0]['avg_accuracy'] - strategy_stats[1]['avg_accuracy']
        if strategy_diff < 0.01:  # Strategies are close
            print(f"   🔍 Strategies perform similarly (±{strategy_diff:.3f}) - focus on BM25 optimization")
        else:
            print(f"   🔍 Clear strategy winner ({strategy_diff:.3f} advantage) - focus on this strategy")
    
    # Save complete results with trend analysis
    output_file = "optimization_results_topic_model.json"
    results_summary = {
        'config': {
            'embedding_models': opt_config.embedding_models,
            'fusion_strategies': opt_config.fusion_strategies,
            'chunk_sizes': opt_config.chunk_sizes,
            'overlap_ratios': opt_config.overlap_ratios,
            'use_condensed_topics': opt_config.use_condensed_topics,
            'optimize_bm25': opt_config.optimize_bm25,
            'evaluation_samples': len(statements),
            'total_configs_tested': len(best_configs),
            'bm25_only_configs': bm25_configs_count,
            'semantic_configs': len(best_configs) - bm25_configs_count
        },
        'best_configurations': best_configs,
        'trend_analysis': {
            'embedding_model_performance': model_stats,
            'fusion_strategy_performance': strategy_stats,
            'chunk_size_analysis': [(size, avg, max_acc, count) for size, avg, max_acc, count in chunk_stats],
            'overlap_ratio_analysis': [(ratio, avg, max_acc, count) for ratio, avg, max_acc, count in overlap_stats],
            'performance_distribution': {
                'overall_mean': float(np.mean(all_accuracies)),
                'overall_std': float(np.std(all_accuracies)),
                'overall_max': float(np.max(all_accuracies)),
                'overall_min': float(np.min(all_accuracies)),
                'p95': float(np.percentile(all_accuracies, 95)),
                'bm25_best': float(np.max(bm25_only_accuracies)) if bm25_only_accuracies else None,
                'semantic_best': float(np.max(semantic_accuracies)) if semantic_accuracies else None,
                'fusion_benefit_pct': float(fusion_benefit) if bm25_only_accuracies and semantic_accuracies else None
            }
        },
        'detailed_results': all_results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Complete results saved to: {output_file}")
    
    # Implementation recommendations
    print("\n🎯 IMPLEMENTATION RECOMMENDATIONS:")
    print("-" * 50)
    
    best = best_configs[0]
    improvement = (best['top1_accuracy'] - 0.895) * 100  # vs current ~89.5%
    print(f"🥇 Best Overall: {best['model'].split('/')[-1]} + {best['strategy']}")
    print(f"   📊 Accuracy: {best['top1_accuracy']:.1%} (+{improvement:.1f}% improvement)")
    print(f"   ⚙️ BM25: chunk_size={best['chunk_size']}, overlap={best['overlap']}")
    print(f"   ⚡ Speed: {best['time_per_query']:.3f}s per query")
    
    # Find best hybrid approach
    hybrid_configs = [c for c in best_configs if c['strategy'] in ['linear_0.3', 'linear_0.5', 'linear_0.7', 'rrf', 'adaptive']]
    if hybrid_configs:
        best_hybrid = hybrid_configs[0]
        print(f"\n🔗 Best Hybrid: {best_hybrid['model'].split('/')[-1]} + {best_hybrid['strategy']}")
        print(f"   📊 Accuracy: {best_hybrid['top1_accuracy']:.1%}")
        print(f"   ⚙️ BM25: chunk_size={best_hybrid['chunk_size']}, overlap={best_hybrid['overlap']}")
    
    # Analyze BM25 optimization impact
    if opt_config.optimize_bm25 and len(opt_config.chunk_sizes) > 1:
        print(f"\n📈 BM25 OPTIMIZATION ANALYSIS:")
        baseline_configs = [c for c in best_configs if c['chunk_size'] == 128]  # Current baseline
        if baseline_configs:
            baseline_acc = max(c['top1_accuracy'] for c in baseline_configs)
            best_bm25_acc = best_configs[0]['top1_accuracy']
            bm25_improvement = (best_bm25_acc - baseline_acc) * 100
            print(f"   🔧 BM25 optimization contributed: +{bm25_improvement:.2f}% accuracy")
    
    print(f"\n✅ Optimization complete! Found {len(best_configs)} configurations.")

if __name__ == "__main__":
    # Ensure we have required dependencies
    try:
        import sentence_transformers
        import sklearn
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install with: pip install sentence-transformers scikit-learn")
        exit(1)
    
    # Run optimization with graceful interrupt handling
    try:
        run_optimization()
    except KeyboardInterrupt:
        print(f"\n\n🛑 OPTIMIZATION INTERRUPTED BY USER")
        print(f"=" * 60)
        print(f"💾 Your progress has been automatically saved!")
        print(f"📂 Look for files: optimization_results_partial_*.json")
        print(f"🔍 To see your results: ls -la optimization_results_partial_*.json")
        print(f"📊 To view best configs so far:")
        print(f"   python -c \"import json; data=json.load(open(sorted([f for f in __import__('os').listdir('.') if f.startswith('optimization_results_partial')])[-1])); print('Configs tested:', data['progress']['total_configs_tested']); print('Best accuracy so far:', max([c['top1_accuracy'] for c in data['progress']['best_configurations']]) if data['progress']['best_configurations'] else 'None')\"")
        print(f"\n✨ You can resume optimization later or use current results!")
        exit(0)
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        print(f"💾 Checking for saved progress...")
        import os
        partial_files = [f for f in os.listdir('.') if f.startswith('optimization_results_partial')]
        if partial_files:
            latest_file = sorted(partial_files)[-1]
            print(f"📂 Latest saved progress: {latest_file}")
        else:
            print(f"❌ No partial results found")
        raise