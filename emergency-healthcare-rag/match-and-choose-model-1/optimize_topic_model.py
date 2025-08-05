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
    
    def __post_init__(self):
        if self.embedding_models is None:
            self.embedding_models = [
                "sentence-transformers/all-MiniLM-L6-v2",       # Fast, lightweight (384d)
                "sentence-transformers/all-mpnet-base-v2",      # Best general performance (768d)
                "sentence-transformers/all-distilroberta-v1",   # Different architecture (768d)
            ]
        
        if self.fusion_strategies is None:
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
                # Comprehensive BM25 optimization (longer runtime)
                self.chunk_sizes = [96, 128, 160]
                self.overlap_ratios = [0.0, 0.1, 0.2]
            else:
                # Use optimized values from separated-models-2 (faster)
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
        print(f"📚 Loading cached BM25 index ({topic_type}, {chunk_size}, {overlap})...")
        try:
            data = pickle.loads(cache_path.read_bytes())
            # Check if it has the expected structure
            if 'chunks' in data and 'topics' in data:
                return data
            else:
                print(f"⚠️ Old cache format detected, rebuilding...")
                cache_path.unlink()  # Delete old cache
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}, rebuilding...")
            cache_path.unlink()  # Delete corrupted cache

    topic_dir = CONDENSED_TOPIC_DIR if use_condensed_topics else ORIGINAL_TOPIC_DIR
    print(f"🔨 Building BM25 index — {topic_type}_topics size={chunk_size} overlap={overlap}")
    
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
        print(f"📚 Loading cached semantic index for {model_name} ({topic_type})...")
        try:
            data = pickle.loads(cache_path.read_bytes())
            # Check if it has the expected structure
            if 'embeddings' in data and 'model_name' in data:
                return data
            else:
                print(f"⚠️ Old cache format detected, rebuilding...")
                cache_path.unlink()  # Delete old cache
        except Exception as e:
            print(f"⚠️ Cache loading failed: {e}, rebuilding...")
            cache_path.unlink()  # Delete corrupted cache

    print(f"🧠 Building semantic index for {model_name} ({topic_type})...")
    
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
# MAIN OPTIMIZATION PIPELINE
# -----------------------------------------------------------------------

def run_optimization():
    """Run comprehensive optimization of topic model"""
    print("🚀 Starting Topic Model Optimization")
    print("=" * 60)
    
    # Configuration - modify these settings as needed
    opt_config = OptimizationConfig(
        optimize_bm25=True,  # Find best BM25 parameters first, then test fusion strategies
        use_condensed_topics=True,  # True for condensed, False for original topics
        max_samples=200,  # Number of statements to evaluate (max 200)
        cache_embeddings=True  # Cache embeddings for faster re-runs
    )
    
    print(f"📋 Configuration:")
    print(f"   🗂️ Topics: {'Condensed' if opt_config.use_condensed_topics else 'Original'}")
    print(f"   🔧 BM25 Optimization: {'Enabled' if opt_config.optimize_bm25 else 'Disabled (using best known config)'}")
    print(f"   📊 Samples: {opt_config.max_samples}")
    print(f"   🧠 Models: {len(opt_config.embedding_models)}")
    print(f"   🔍 Strategies: {len(opt_config.fusion_strategies)}")
    
    total_configs = len(opt_config.chunk_sizes) * len(opt_config.overlap_ratios) * len(opt_config.embedding_models) * len(opt_config.fusion_strategies)
    estimated_time = total_configs * 2  # Rough estimate: 2 minutes per config
    print(f"   ⏱️ Estimated runtime: {estimated_time//60:.0f}-{estimated_time//60*2:.0f} minutes ({total_configs} configurations)")
    
    # Load test data
    print("📚 Loading evaluation data...")
    statements = load_statements()
    if opt_config.max_samples and opt_config.max_samples < len(statements):
        statements = statements[:opt_config.max_samples]
    print(f"📊 Evaluating on {len(statements)} statements")
    
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
    
    # Test each BM25 configuration
    for bm25_idx, (chunk_size, overlap) in enumerate(bm25_configs):
        print(f"\n🔨 BM25 Config {bm25_idx+1}/{len(bm25_configs)}: chunk_size={chunk_size}, overlap={overlap}")
        
        # Build BM25 index for this configuration
        bm25_data = build_bm25_index(chunk_size, overlap, opt_config.use_condensed_topics)
        print(f"   ✅ BM25 index ready with {len(bm25_data['chunks'])} chunks")
        
        # Test each embedding model with this BM25 config
        for model_name in opt_config.embedding_models:
            print(f"\n🧠 Processing model: {model_name}")
            
            # Build semantic index
            semantic_data = build_semantic_index(model_name, bm25_data, opt_config.cache_embeddings)
            if semantic_data is None:
                print(f"   ❌ Skipping {model_name} due to loading error")
                continue
            
            # Create searcher
            searcher = HybridSearcher(bm25_data, semantic_data)
            
            # Test each fusion strategy
            for strategy in opt_config.fusion_strategies:
                config_key = f"{model_name}_{strategy}_cs{chunk_size}_ov{overlap}"
                
                # Skip semantic strategies if no semantic model
                if strategy == "semantic_only" and semantic_data is None:
                    continue
                
                # Run evaluation
                start_time = time.time()
                eval_results = evaluate_search_strategy(searcher, strategy, statements, opt_config)
                elapsed = time.time() - start_time
                
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
                print(f"   🔍 {strategy}: T1: {metrics['top1_accuracy']:.3f}, "
                      f"T2: {metrics['top2_accuracy']:.3f}, "
                      f"T3: {metrics['top3_accuracy']:.3f}, "
                      f"T5: {metrics['top5_accuracy']:.3f}, "
                      f"MRR: {metrics['mrr']:.3f}")
    
    # Find and display best configurations
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION RESULTS")
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
    
    # Save complete results
    output_file = "optimization_results_topic_model.json"
    results_summary = {
        'config': {
            'embedding_models': opt_config.embedding_models,
            'fusion_strategies': opt_config.fusion_strategies,
            'chunk_sizes': opt_config.chunk_sizes,
            'overlap_ratios': opt_config.overlap_ratios,
            'use_condensed_topics': opt_config.use_condensed_topics,
            'optimize_bm25': opt_config.optimize_bm25,
            'evaluation_samples': len(statements)
        },
        'best_configurations': best_configs,
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
    
    run_optimization()