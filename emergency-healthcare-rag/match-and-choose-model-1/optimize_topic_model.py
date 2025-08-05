#!/usr/bin/env python3
"""
Optimize topic model with hybrid BM25 + semantic search approaches
Tests multiple embedding models and fusion strategies to find complementary models
"""

import os
import json
import pickle
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass

# Core dependencies
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# ----------------------------------------------------------------------- 
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Configuration for optimization runs"""
    # Models to test (will be downloaded automatically)
    embedding_models = [
        "sentence-transformers/all-MiniLM-L6-v2",       # Fast, lightweight (384d)
        "sentence-transformers/all-mpnet-base-v2",      # Best general performance (768d)
        "sentence-transformers/all-distilroberta-v1",   # Different architecture (768d)
    ]
    
    # Fusion strategies to test
    fusion_strategies = [
        "bm25_only",           # Baseline
        "semantic_only",       # Semantic baseline
        "linear_0.3",          # 0.3*BM25 + 0.7*semantic
        "linear_0.5",          # 0.5*BM25 + 0.5*semantic  
        "linear_0.7",          # 0.7*BM25 + 0.3*semantic
        "rrf",                 # Reciprocal Rank Fusion
        "adaptive"             # Adaptive weighting
    ]
    
    # BM25 chunk parameters (optimized from separated-models-2)
    chunk_size: int = 128
    overlap: int = 12
    
    # Search parameters
    top_k: int = 10
    
    # Evaluation parameters
    max_samples: int = 200  # Use full train set for comprehensive evaluation
    cache_embeddings: bool = True
    save_detailed_results: bool = True

# Global paths
CONDENSED_TOPIC_DIR = Path("data/condensed_topics")
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
    top3_correct = sum(1 for r in results if r['rank_correct'] <= 3)
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
        'top3_accuracy': top3_correct / total,  
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

def build_bm25_index(config: OptimizationConfig) -> Dict:
    """Build BM25 index from condensed topics"""
    cache_path = CACHE_ROOT / f"bm25_index_opt_{config.chunk_size}_{config.overlap}.pkl"
    
    if cache_path.exists():
        print(f"📚 Loading cached BM25 index...")
        return pickle.loads(cache_path.read_bytes())

    print(f"🔨 Building BM25 index (chunk_size={config.chunk_size}, overlap={config.overlap})...")
    
    chunks = []
    topics = []
    topic_names = []
    
    for md_file in tqdm(CONDENSED_TOPIC_DIR.rglob("*.md"), desc="Processing topics"):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP.get(topic_name, -1)
        
        if topic_id == -1:
            continue
            
        words = md_file.read_text(encoding="utf-8").split()
        for chunk_text in chunk_words(words, config.chunk_size, config.overlap):
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
        'config': config
    }
    
    cache_path.write_bytes(pickle.dumps(data))
    print(f"💾 Cached BM25 index to {cache_path}")
    return data

# -----------------------------------------------------------------------
# SEMANTIC EMBEDDING BUILDING
# -----------------------------------------------------------------------

def build_semantic_index(model_name: str, bm25_data: Dict, config: OptimizationConfig) -> Dict:
    """Build semantic embeddings for all chunks"""
    model_slug = model_name.replace("/", "_").replace("-", "_")
    cache_path = CACHE_ROOT / f"semantic_index_{model_slug}_{config.chunk_size}_{config.overlap}.pkl"
    
    if cache_path.exists() and config.cache_embeddings:
        print(f"📚 Loading cached semantic index for {model_name}...")
        return pickle.loads(cache_path.read_bytes())

    print(f"🧠 Building semantic index for {model_name}...")
    
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
    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding batches"):
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
        'config': config
    }
    
    if config.cache_embeddings:
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
    
    config = OptimizationConfig()
    
    # Load test data
    print("📚 Loading evaluation data...")
    statements = load_statements()
    if config.max_samples and config.max_samples < len(statements):
        statements = statements[:config.max_samples]
    print(f"📊 Evaluating on {len(statements)} statements")
    
    # Build BM25 index
    print("\n🔨 Building BM25 index...")
    bm25_data = build_bm25_index(config)
    print(f"✅ BM25 index ready with {len(bm25_data['chunks'])} chunks")
    
    # Store all results
    all_results = {}
    
    # Test each embedding model
    for model_name in config.embedding_models:
        print(f"\n🧠 Processing model: {model_name}")
        print("-" * 40)
        
        # Build semantic index
        semantic_data = build_semantic_index(model_name, bm25_data, config)
        if semantic_data is None:
            print(f"❌ Skipping {model_name} due to loading error")
            continue
        
        # Create searcher
        searcher = HybridSearcher(bm25_data, semantic_data)
        
        # Test each fusion strategy
        model_results = {}
        for strategy in config.fusion_strategies:
            print(f"🔍 Testing strategy: {strategy}")
            
            # Skip semantic strategies if no semantic model
            if strategy == "semantic_only" and semantic_data is None:
                continue
            
            # Run evaluation
            start_time = time.time()
            eval_results = evaluate_search_strategy(searcher, strategy, statements, config)
            elapsed = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_metrics(eval_results)
            metrics['evaluation_time'] = elapsed
            metrics['time_per_query'] = elapsed / len(statements)
            
            model_results[strategy] = {
                'metrics': metrics,
                'detailed_results': eval_results if config.save_detailed_results else None
            }
            
            # Print summary
            print(f"  📈 Top-1: {metrics['top1_accuracy']:.3f}, "
                  f"Top-3: {metrics['top3_accuracy']:.3f}, "
                  f"MRR: {metrics['mrr']:.3f}, "
                  f"Avg Sep: {metrics['avg_separation_when_correct']:.2f}")
        
        all_results[model_name] = model_results
    
    # Find best configurations
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    best_configs = []
    for model_name, model_results in all_results.items():
        for strategy, results in model_results.items():
            metrics = results['metrics']
            best_configs.append({
                'model': model_name,
                'strategy': strategy,
                'top1_accuracy': metrics['top1_accuracy'],
                'top3_accuracy': metrics['top3_accuracy'],
                'mrr': metrics['mrr'],
                'avg_separation': metrics['avg_separation_when_correct'],
                'score_gap_p90': metrics['score_gaps']['percentiles']['p90'],
                'time_per_query': metrics['time_per_query']
            })
    
    # Sort by top-1 accuracy first, then MRR
    best_configs.sort(key=lambda x: (x['top1_accuracy'], x['mrr']), reverse=True)
    
    print("\n🥇 TOP 10 CONFIGURATIONS:")
    print(f"{'Rank':<4} {'Model':<35} {'Strategy':<15} {'Top-1':<6} {'Top-3':<6} {'MRR':<6} {'Sep':<6} {'Gap90':<6} {'Time':<6}")
    print("-" * 100)
    
    for i, config in enumerate(best_configs[:10], 1):
        model_short = config['model'].split('/')[-1][:30]
        print(f"{i:<4} {model_short:<35} {config['strategy']:<15} "
              f"{config['top1_accuracy']:.3f}  {config['top3_accuracy']:.3f}  "
              f"{config['mrr']:.3f}  {config['avg_separation']:.2f}   "
              f"{config['score_gap_p90']:.2f}   {config['time_per_query']:.3f}")
    
    # Save complete results
    output_file = "optimization_results_topic_model.json"
    results_summary = {
        'config': {
            'embedding_models': config.embedding_models,
            'fusion_strategies': config.fusion_strategies,
            'chunk_size': config.chunk_size,
            'overlap': config.overlap,
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
    print("-" * 40)
    
    best = best_configs[0]
    print(f"🥇 Best Overall: {best['model']} with {best['strategy']}")
    print(f"   📊 Accuracy: {best['top1_accuracy']:.1%} (vs current ~89.5%)")
    print(f"   ⚡ Speed: {best['time_per_query']:.3f}s per query")
    
    # Find best complementary pair
    semantic_configs = [c for c in best_configs if 'semantic' in c['strategy'] or 'linear' in c['strategy'] or 'rrf' in c['strategy']]
    if semantic_configs:
        best_hybrid = semantic_configs[0]
        print(f"\n🔗 Best Hybrid: {best_hybrid['model']} with {best_hybrid['strategy']}")
        print(f"   📊 Accuracy: {best_hybrid['top1_accuracy']:.1%}")
        print(f"   🔍 Strategy: Combines BM25 + semantic search")
    
    # Threshold recommendations
    gap_analysis = []
    for model_results in all_results.values():
        for strategy_results in model_results.values():
            if strategy_results['metrics']['score_gaps']['percentiles']:
                gaps = strategy_results['metrics']['score_gaps']['percentiles']
                gap_analysis.append(gaps)
    
    if gap_analysis:
        avg_gaps = {k: np.mean([g[k] for g in gap_analysis]) for k in gap_analysis[0].keys()}
        print(f"\n🎚️ Recommended Thresholds (based on score gap analysis):")
        print(f"   Conservative (high precision): {avg_gaps['p75']:.2f}")
        print(f"   Balanced: {avg_gaps['p50']:.2f}")
        print(f"   Aggressive (high recall): {avg_gaps['p25']:.2f}")
    
    print(f"\n✅ Optimization complete! Check {output_file} for detailed results.")

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