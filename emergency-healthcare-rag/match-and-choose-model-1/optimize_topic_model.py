#!/usr/bin/env python3
"""
Optimized Topic Model with Hierarchical Approach

Phase 1: Extensive BM25 optimization by itself
Phase 2: Find best embedding model with its hyperparameters  
Phase 3: Combine top 5 most different configs from each
Phase 4: Zoom into promising combinations

Never uses online API calls - downloads models to local.
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
import itertools
from collections import defaultdict

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
    tqdm.monitor_interval = 0
    USE_TQDM = True
except ImportError:
    USE_TQDM = False
    def tqdm(iterable, desc="", leave=True, **kwargs):
        total = len(iterable) if hasattr(iterable, '__len__') else None
        for i, item in enumerate(iterable):
            if total and i % max(1, total // 20) == 0:
                print(f"{desc}: {i+1}/{total} ({100*(i+1)/total:.0f}%)")
            yield item

# ----------------------------------------------------------------------- 
# CONFIGURATION & SETUP
# -----------------------------------------------------------------------

@dataclass
class OptimizationConfig:
    """Configuration for hierarchical optimization"""
    
    # Phase 1: BM25 optimization parameters
    bm25_chunk_sizes: List[int] = None
    bm25_overlap_ratios: List[float] = None
    
    # Phase 2: Embedding model parameters  
    embedding_models: List[str] = None
    embedding_chunk_sizes: List[int] = None
    embedding_overlap_ratios: List[float] = None
    
    # Phase 3: Combination parameters
    top_bm25_configs: int = 5
    top_embedding_configs: int = 5
    fusion_strategies: List[str] = None
    
    # Phase 4: Zoom parameters
    zoom_enabled: bool = True
    zoom_radius: int = 2  # Parameter variations around promising configs
    
    # General parameters
    use_condensed_topics: bool = True
    max_samples: int = 50  # Reduced from 200 for faster testing
    cache_embeddings: bool = True
    save_detailed_results: bool = True
    
    def __post_init__(self):
        if self.bm25_chunk_sizes is None:
            # Extensive BM25 optimization - reduced for faster testing
            self.bm25_chunk_sizes = [96, 112, 128, 144, 160]  # Reduced from 11 to 5
            
        if self.bm25_overlap_ratios is None:
            # Use no overlap for now to get the optimization working
            self.bm25_overlap_ratios = [0]  # No overlap for now
            
        if self.embedding_models is None:
            # Models that will be downloaded locally - reduced for faster testing
            self.embedding_models = [
                "sentence-transformers/all-MiniLM-L6-v2",       # Fast, lightweight (384d)
                "sentence-transformers/all-mpnet-base-v2",      # Best general performance (768d)
            ]  # Reduced from 5 to 2
            
        if self.embedding_chunk_sizes is None:
            # Focus on promising ranges for embeddings - reduced for faster testing
            self.embedding_chunk_sizes = [112, 128, 144]  # Reduced from 5 to 3
            
        if self.embedding_overlap_ratios is None:
            # Use no overlap for now
            self.embedding_overlap_ratios = [0]  # No overlap for now
            
        if self.fusion_strategies is None:
            self.fusion_strategies = [
                "bm25_only",           # Always include baseline
                "semantic_only",       # Pure semantic baseline
                "linear_0.5",          # Balanced fusion
                "rrf",                 # Reciprocal Rank Fusion
            ]  # Reduced from 6 to 4

# Global configuration
config = OptimizationConfig()

# ----------------------------------------------------------------------- 
# UTILITY FUNCTIONS
# -----------------------------------------------------------------------

def chunk_words(words: List[str], size: int, overlap: int) -> List[str]:
    """Split words into overlapping chunks"""
    if len(words) <= size:
        return [' '.join(words)]
    
    # Use a more efficient approach - step by (size - overlap) instead of creating all possible overlaps
    step_size = max(1, size - overlap)
    
    chunks = []
    for i in range(0, len(words) - size + 1, step_size):
        chunk = words[i:i + size]
        chunks.append(' '.join(chunk))
    
    # If we didn't get any chunks (due to step size being too large), create at least one chunk
    if not chunks:
        chunks = [' '.join(words[:size])]
    
    return chunks

def load_statements() -> List[Tuple[str, int]]:
    """Load training statements with true topics"""
    statement_dir = Path("data/train/statements")
    answer_dir = Path("data/train/answers")
    
    records = []
    for path in sorted(statement_dir.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((answer_dir / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    
    return records[:config.max_samples]

def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive metrics from results"""
    if not results:
        return {"accuracy": 0.0, "avg_rank": 0.0, "top3_accuracy": 0.0}
    
    total = len(results)
    correct = sum(1 for r in results if r['rank_correct'] == 1)
    top3_correct = sum(1 for r in results if r['rank_correct'] <= 3)
    avg_rank = sum(r['rank_correct'] for r in results) / total
    
    return {
        "accuracy": correct / total,
        "avg_rank": avg_rank,
        "top3_accuracy": top3_correct / total,
        "total_samples": total
    }

# ----------------------------------------------------------------------- 
# PHASE 1: BM25 OPTIMIZATION
# -----------------------------------------------------------------------

def build_bm25_index(chunk_size: int, overlap: int, use_condensed_topics: bool = True) -> Dict:
    """Build BM25 index with given parameters"""
    topic_dir = Path("data/condensed_topics" if use_condensed_topics else "data/topics")
    topic_map = json.loads(Path("data/topics.json").read_text())
    
    documents = []
    doc_ids = []
    
    for topic_name, topic_id in topic_map.items():
        if use_condensed_topics:
            # For condensed topics, look for directory with topic name
            topic_path = topic_dir / topic_name
            if topic_path.exists() and topic_path.is_dir():
                # Read all .md files in the directory
                for md_file in topic_path.glob("*.md"):
                    content = md_file.read_text()
                    words = content.split()
                    chunks = chunk_words(words, chunk_size, int(chunk_size * overlap))
                    
                    for chunk in chunks:
                        documents.append(chunk)
                        doc_ids.append(topic_id)
        else:
            # For original topics, look for individual .md file
            topic_path = topic_dir / f"{topic_id}.md"
            if topic_path.exists():
                content = topic_path.read_text()
                words = content.split()
                chunks = chunk_words(words, chunk_size, int(chunk_size * overlap))
                
                for chunk in chunks:
                    documents.append(chunk)
                    doc_ids.append(topic_id)
    
    # Check if we have any documents
    if not documents:
        raise ValueError(f"No documents found for chunk_size={chunk_size}, overlap={overlap}")
    
    print(f"   Built index with {len(documents)} chunks from {len(set(doc_ids))} topics")
    
    # Build BM25 index
    tokenized_docs = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    return {
        "bm25": bm25,
        "documents": documents,
        "doc_ids": doc_ids,
        "chunk_size": chunk_size,
        "overlap": overlap
    }

def evaluate_bm25_config(bm25_data: Dict, statements: List[Tuple[str, int]]) -> Dict:
    """Evaluate a single BM25 configuration"""
    bm25 = bm25_data["bm25"]
    doc_ids = bm25_data["doc_ids"]
    
    results = []
    for statement, true_topic in tqdm(statements, desc="BM25 evaluation", disable=not USE_TQDM):
        try:
            # Search
            tokenized_query = statement.split()
            scores = bm25.get_scores(tokenized_query)
            
            # Check if we have valid scores
            if len(scores) == 0 or np.all(scores == 0):
                print(f"   ⚠️ Warning: No valid scores for statement: {statement[:50]}...")
                continue
            
            # Get top results
            top_indices = np.argsort(scores)[::-1][:10]
            top_topics = [doc_ids[i] for i in top_indices]
            
            # Find rank of correct topic
            rank_correct = 0
            for rank, topic_id in enumerate(top_topics, 1):
                if topic_id == true_topic:
                    rank_correct = rank
                    break
            
            results.append({
                'statement': statement,
                'true_topic': true_topic,
                'rank_correct': rank_correct,
                'top_topics': top_topics[:5]
            })
            
        except Exception as e:
            print(f"   ⚠️ Error processing statement: {e}")
            continue
    
    if not results:
        raise ValueError("No valid results generated for BM25 evaluation")
    
    metrics = calculate_metrics(results)
    return {
        "config": {
            "chunk_size": bm25_data["chunk_size"],
            "overlap": bm25_data["overlap"],
            "strategy": "bm25_only"
        },
        "metrics": metrics,
        "results": results
    }

def run_bm25_optimization() -> List[Dict]:
    """Phase 1: Extensive BM25 optimization"""
    print("🔍 PHASE 1: Extensive BM25 Optimization")
    print(f"Testing {len(config.bm25_chunk_sizes)} chunk sizes × {len(config.bm25_overlap_ratios)} overlap ratios = {len(config.bm25_chunk_sizes) * len(config.bm25_overlap_ratios)} configurations")
    print("   (Reduced configuration space for faster testing)")
    
    statements = load_statements()
    all_results = []
    
    total_configs = len(config.bm25_chunk_sizes) * len(config.bm25_overlap_ratios)
    config_count = 0
    
    # Initialize live results
    update_live_results(all_results, "phase1_bm25", status="starting")
    
    for chunk_size in config.bm25_chunk_sizes:
        for overlap_ratio in config.bm25_overlap_ratios:
            config_count += 1
            print(f"\n📊 BM25 Config {config_count}/{total_configs}: chunk_size={chunk_size}, overlap={overlap_ratio} words")
            
            # Update live results with current config
            current_config = {"chunk_size": chunk_size, "overlap": overlap_ratio, "config_count": config_count, "total_configs": total_configs}
            update_live_results(all_results, "phase1_bm25", current_config, "running")
            
            try:
                bm25_data = build_bm25_index(chunk_size, overlap_ratio, config.use_condensed_topics)
                result = evaluate_bm25_config(bm25_data, statements)
                all_results.append(result)
                
                print(f"✅ Accuracy: {result['metrics']['accuracy']:.3f}, Avg Rank: {result['metrics']['avg_rank']:.2f}")
                
                # Update live results after each successful config
                update_live_results(all_results, "phase1_bm25", current_config, "running")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    # Sort by accuracy and select top configs
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    top_results = all_results[:config.top_bm25_configs]
    
    print(f"\n🏆 TOP {config.top_bm25_configs} BM25 CONFIGURATIONS:")
    for i, result in enumerate(top_results, 1):
        cfg = result['config']
        metrics = result['metrics']
        print(f"{i}. chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']:.2f} → Accuracy: {metrics['accuracy']:.3f}")
    
    # Update live results with final results
    update_live_results(all_results, "phase1_bm25", status="completed")
    
    return top_results

# ----------------------------------------------------------------------- 
# PHASE 2: EMBEDDING MODEL OPTIMIZATION
# -----------------------------------------------------------------------

def build_semantic_index(model_name: str, chunk_size: int, overlap: int, use_condensed_topics: bool = True) -> Dict:
    """Build semantic index with given parameters"""
    print(f"📥 Downloading model: {model_name}")
    
    # Download model locally (no online API calls)
    try:
        model = SentenceTransformer(model_name, device='cpu')
    except Exception as e:
        print(f"   ❌ Failed to download model {model_name}: {e}")
        raise
    
    topic_dir = Path("data/condensed_topics" if use_condensed_topics else "data/topics")
    topic_map = json.loads(Path("data/topics.json").read_text())
    
    documents = []
    doc_ids = []
    
    for topic_name, topic_id in topic_map.items():
        if use_condensed_topics:
            # For condensed topics, look for directory with topic name
            topic_path = topic_dir / topic_name
            if topic_path.exists() and topic_path.is_dir():
                # Read all .md files in the directory
                for md_file in topic_path.glob("*.md"):
                    content = md_file.read_text()
                    words = content.split()
                    chunks = chunk_words(words, chunk_size, int(chunk_size * overlap))
                    
                    for chunk in chunks:
                        documents.append(chunk)
                        doc_ids.append(topic_id)
        else:
            # For original topics, look for individual .md file
            topic_path = topic_dir / f"{topic_id}.md"
            if topic_path.exists():
                content = topic_path.read_text()
                words = content.split()
                chunks = chunk_words(words, chunk_size, int(chunk_size * overlap))
                
                for chunk in chunks:
                    documents.append(chunk)
                    doc_ids.append(topic_id)
    
    # Check if we have any documents
    if not documents:
        raise ValueError(f"No documents found for chunk_size={chunk_size}, overlap={overlap}")
    
    print(f"   Built index with {len(documents)} chunks from {len(set(doc_ids))} topics")
    
    # Generate embeddings
    print(f"🔧 Generating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=USE_TQDM)
    
    return {
        "model": model,
        "embeddings": embeddings,
        "documents": documents,
        "doc_ids": doc_ids,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name
    }

def evaluate_semantic_config(semantic_data: Dict, statements: List[Tuple[str, int]]) -> Dict:
    """Evaluate a single semantic configuration"""
    model = semantic_data["model"]
    embeddings = semantic_data["embeddings"]
    doc_ids = semantic_data["doc_ids"]
    
    results = []
    for statement, true_topic in tqdm(statements, desc="Semantic evaluation", disable=not USE_TQDM):
        try:
            # Generate query embedding
            query_embedding = model.encode([statement])
            
            # Check if embeddings array is not empty
            if len(embeddings) == 0:
                print(f"   ⚠️ Warning: Empty embeddings array for statement: {statement[:50]}...")
                continue
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:10]
            top_topics = [doc_ids[i] for i in top_indices]
            
            # Find rank of correct topic
            rank_correct = 0
            for rank, topic_id in enumerate(top_topics, 1):
                if topic_id == true_topic:
                    rank_correct = rank
                    break
            
            results.append({
                'statement': statement,
                'true_topic': true_topic,
                'rank_correct': rank_correct,
                'top_topics': top_topics[:5]
            })
            
        except Exception as e:
            print(f"   ⚠️ Error processing statement: {e}")
            continue
    
    if not results:
        raise ValueError("No valid results generated for semantic evaluation")
    
    metrics = calculate_metrics(results)
    return {
        "config": {
            "model_name": semantic_data["model_name"],
            "chunk_size": semantic_data["chunk_size"],
            "overlap": semantic_data["overlap"],
            "strategy": "semantic_only"
        },
        "metrics": metrics,
        "results": results
    }

def run_embedding_optimization() -> List[Dict]:
    """Phase 2: Embedding model optimization"""
    print("\n🔍 PHASE 2: Embedding Model Optimization")
    print(f"Testing {len(config.embedding_models)} models × {len(config.embedding_chunk_sizes)} chunk sizes × {len(config.embedding_overlap_ratios)} overlap ratios")
    print("   (Reduced configuration space for faster testing)")
    
    statements = load_statements()
    all_results = []
    
    total_configs = len(config.embedding_models) * len(config.embedding_chunk_sizes) * len(config.embedding_overlap_ratios)
    config_count = 0
    
    # Initialize live results
    update_live_results(all_results, "phase2_semantic", status="starting")
    
    for model_name in config.embedding_models:
        for chunk_size in config.embedding_chunk_sizes:
            for overlap_ratio in config.embedding_overlap_ratios:
                config_count += 1
                print(f"\n📊 Embedding Config {config_count}/{total_configs}: {model_name}, chunk_size={chunk_size}, overlap={overlap_ratio} words")
                
                # Update live results with current config
                current_config = {"model_name": model_name, "chunk_size": chunk_size, "overlap": overlap_ratio, "config_count": config_count, "total_configs": total_configs}
                update_live_results(all_results, "phase2_semantic", current_config, "running")
                
                try:
                    semantic_data = build_semantic_index(model_name, chunk_size, overlap_ratio, config.use_condensed_topics)
                    result = evaluate_semantic_config(semantic_data, statements)
                    all_results.append(result)
                    
                    print(f"✅ Accuracy: {result['metrics']['accuracy']:.3f}, Avg Rank: {result['metrics']['avg_rank']:.2f}")
                    
                    # Update live results after each successful config
                    update_live_results(all_results, "phase2_semantic", current_config, "running")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
    
    # Sort by accuracy and select top configs
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    top_results = all_results[:config.top_embedding_configs]
    
    print(f"\n🏆 TOP {config.top_embedding_configs} EMBEDDING CONFIGURATIONS:")
    for i, result in enumerate(top_results, 1):
        cfg = result['config']
        metrics = result['metrics']
        print(f"{i}. {cfg['model_name']}, chunk_size={cfg['chunk_size']}, overlap={cfg['overlap']:.2f} → Accuracy: {metrics['accuracy']:.3f}")
    
    # Update live results with final results
    update_live_results(all_results, "phase2_semantic", status="completed")
    
    return top_results

# ----------------------------------------------------------------------- 
# PHASE 3: HYBRID COMBINATIONS
# -----------------------------------------------------------------------

class HybridSearcher:
    """Hybrid search combining BM25 and semantic approaches"""
    
    def __init__(self, bm25_data: Dict, semantic_data: Dict):
        self.bm25_data = bm25_data
        self.semantic_data = semantic_data
        self.bm25 = bm25_data["bm25"]
        self.semantic_model = semantic_data["model"]
        self.semantic_embeddings = semantic_data["embeddings"]
        self.doc_ids = bm25_data["doc_ids"]  # Should be same for both
    
    def search_linear_fusion(self, query: str, bm25_weight: float = 0.5, top_k: int = 10) -> List[Dict]:
        """Linear fusion of BM25 and semantic scores"""
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Semantic search
        query_embedding = self.semantic_model.encode([query])
        semantic_similarities = cosine_similarity(query_embedding, self.semantic_embeddings)[0]
        
        # Normalize scores to [0,1] range
        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        semantic_scores_norm = (semantic_similarities - semantic_similarities.min()) / (semantic_similarities.max() - semantic_similarities.min() + 1e-8)
        
        # Combine scores
        combined_scores = bm25_weight * bm25_scores_norm + (1 - bm25_weight) * semantic_scores_norm
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        return [{"topic_id": self.doc_ids[i], "score": combined_scores[i]} for i in top_indices]
    
    def search_rrf(self, query: str, top_k: int = 10, k_param: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion"""
        # BM25 search
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        # Semantic search
        query_embedding = self.semantic_model.encode([query])
        semantic_similarities = cosine_similarity(query_embedding, self.semantic_embeddings)[0]
        semantic_ranks = np.argsort(semantic_similarities)[::-1]
        
        # Calculate RRF scores
        rrf_scores = {}
        for rank, doc_idx in enumerate(bm25_ranks):
            topic_id = self.doc_ids[doc_idx]
            rrf_scores[topic_id] = rrf_scores.get(topic_id, 0) + 1 / (k_param + rank + 1)
        
        for rank, doc_idx in enumerate(semantic_ranks):
            topic_id = self.doc_ids[doc_idx]
            rrf_scores[topic_id] = rrf_scores.get(topic_id, 0) + 1 / (k_param + rank + 1)
        
        # Sort by RRF scores
        sorted_topics = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"topic_id": topic_id, "score": score} for topic_id, score in sorted_topics]

def evaluate_hybrid_config(bm25_config: Dict, semantic_config: Dict, fusion_strategy: str, statements: List[Tuple[str, int]]) -> Dict:
    """Evaluate a hybrid configuration"""
    # Build indices
    bm25_data = build_bm25_index(
        bm25_config['chunk_size'], 
        int(bm25_config['overlap']), 
        config.use_condensed_topics
    )
    
    semantic_data = build_semantic_index(
        semantic_config['model_name'],
        semantic_config['chunk_size'],
        int(semantic_config['overlap']),
        config.use_condensed_topics
    )
    
    # Create hybrid searcher
    searcher = HybridSearcher(bm25_data, semantic_data)
    
    results = []
    for statement, true_topic in tqdm(statements, desc="Hybrid evaluation", disable=not USE_TQDM):
        # Search based on strategy
        if fusion_strategy == "bm25_only":
            tokenized_query = statement.split()
            scores = bm25_data["bm25"].get_scores(tokenized_query)
            top_indices = np.argsort(scores)[::-1][:10]
            top_topics = [bm25_data["doc_ids"][i] for i in top_indices]
        elif fusion_strategy == "semantic_only":
            query_embedding = semantic_data["model"].encode([statement])
            similarities = cosine_similarity(query_embedding, semantic_data["embeddings"])[0]
            top_indices = np.argsort(similarities)[::-1][:10]
            top_topics = [semantic_data["doc_ids"][i] for i in top_indices]
        elif fusion_strategy.startswith("linear_"):
            weight = float(fusion_strategy.split("_")[1])
            search_results = searcher.search_linear_fusion(statement, bm25_weight=weight, top_k=10)
            top_topics = [r["topic_id"] for r in search_results]
        elif fusion_strategy == "rrf":
            search_results = searcher.search_rrf(statement, top_k=10)
            top_topics = [r["topic_id"] for r in search_results]
        else:
            continue
        
        # Find rank of correct topic
        rank_correct = 0
        for rank, topic_id in enumerate(top_topics, 1):
            if topic_id == true_topic:
                rank_correct = rank
                break
        
        results.append({
            'statement': statement,
            'true_topic': true_topic,
            'rank_correct': rank_correct,
            'top_topics': top_topics[:5]
        })
    
    metrics = calculate_metrics(results)
    return {
        "config": {
            "bm25_config": bm25_config,
            "semantic_config": semantic_config,
            "fusion_strategy": fusion_strategy
        },
        "metrics": metrics,
        "results": results
    }

def run_hybrid_optimization(bm25_results: List[Dict], semantic_results: List[Dict]) -> List[Dict]:
    """Phase 3: Combine top configurations"""
    print("\n🔍 PHASE 3: Hybrid Combination Optimization")
    
    statements = load_statements()
    all_results = []
    
    total_combinations = len(bm25_results) * len(semantic_results) * len(config.fusion_strategies)
    combination_count = 0
    
    # Initialize live results
    update_live_results(all_results, "phase3_hybrid", status="starting")
    
    for bm25_result in bm25_results:
        bm25_config = bm25_result['config']
        for semantic_result in semantic_results:
            semantic_config = semantic_result['config']
            for fusion_strategy in config.fusion_strategies:
                combination_count += 1
                print(f"\n📊 Hybrid Config {combination_count}/{total_combinations}: {fusion_strategy}")
                print(f"   BM25: chunk_size={bm25_config['chunk_size']}, overlap={bm25_config['overlap']:.2f}")
                print(f"   Semantic: {semantic_config['model_name']}, chunk_size={semantic_config['chunk_size']}, overlap={semantic_config['overlap']:.2f}")
                
                # Update live results with current config
                current_config = {
                    "fusion_strategy": fusion_strategy,
                    "bm25_config": bm25_config,
                    "semantic_config": semantic_config,
                    "combination_count": combination_count,
                    "total_combinations": total_combinations
                }
                update_live_results(all_results, "phase3_hybrid", current_config, "running")
                
                try:
                    result = evaluate_hybrid_config(bm25_config, semantic_config, fusion_strategy, statements)
                    all_results.append(result)
                    
                    print(f"✅ Accuracy: {result['metrics']['accuracy']:.3f}, Avg Rank: {result['metrics']['avg_rank']:.2f}")
                    
                    # Update live results after each successful config
                    update_live_results(all_results, "phase3_hybrid", current_config, "running")
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
                    continue
    
    # Sort by accuracy
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    
    print(f"\n🏆 TOP HYBRID CONFIGURATIONS:")
    for i, result in enumerate(all_results[:10], 1):
        cfg = result['config']
        metrics = result['metrics']
        print(f"{i}. {cfg['fusion_strategy']} → Accuracy: {metrics['accuracy']:.3f}")
        print(f"   BM25: chunk_size={cfg['bm25_config']['chunk_size']}, overlap={cfg['bm25_config']['overlap']:.2f}")
        print(f"   Semantic: {cfg['semantic_config']['model_name']}, chunk_size={cfg['semantic_config']['chunk_size']}")
    
    # Update live results with final results
    update_live_results(all_results, "phase3_hybrid", status="completed")
    
    return all_results

# ----------------------------------------------------------------------- 
# PHASE 4: ZOOM INTO PROMISING CONFIGURATIONS
# -----------------------------------------------------------------------

def generate_zoom_configs(promising_config: Dict, radius: int = 2) -> List[Dict]:
    """Generate parameter variations around a promising configuration"""
    base_config = promising_config['config']
    zoom_configs = []
    
    # Generate variations around BM25 parameters
    base_chunk_size = base_config['bm25_config']['chunk_size']
    base_overlap = base_config['bm25_config']['overlap']
    
    for chunk_offset in range(-radius, radius + 1):
        for overlap_offset in range(-radius, radius + 1):
            new_chunk_size = base_chunk_size + chunk_offset * 8  # Step by 8
            new_overlap = base_overlap + overlap_offset * 0.05   # Step by 0.05
            
            if new_chunk_size >= 64 and new_chunk_size <= 256 and new_overlap >= 0 and new_overlap <= 0.4:
                zoom_config = base_config.copy()
                zoom_config['bm25_config'] = zoom_config['bm25_config'].copy()
                zoom_config['bm25_config']['chunk_size'] = new_chunk_size
                zoom_config['bm25_config']['overlap'] = new_overlap
                zoom_configs.append(zoom_config)
    
    return zoom_configs

def run_zoom_optimization(promising_results: List[Dict]) -> List[Dict]:
    """Phase 4: Zoom into promising configurations"""
    if not config.zoom_enabled or not promising_results:
        return []
    
    print("\n🔍 PHASE 4: Zoom into Promising Configurations")
    
    statements = load_statements()
    zoom_results = []
    
    # Select top 3 most promising configurations for zoom
    top_configs = promising_results[:3]
    
    for i, promising_result in enumerate(top_configs, 1):
        print(f"\n🔬 Zooming into promising config {i}: {promising_result['config']['fusion_strategy']}")
        print(f"   Current accuracy: {promising_result['metrics']['accuracy']:.3f}")
        
        zoom_configs = generate_zoom_configs(promising_result, config.zoom_radius)
        print(f"   Testing {len(zoom_configs)} parameter variations...")
        
        for j, zoom_config in enumerate(zoom_configs):
            try:
                result = evaluate_hybrid_config(
                    zoom_config['bm25_config'],
                    zoom_config['semantic_config'],
                    zoom_config['fusion_strategy'],
                    statements
                )
                zoom_results.append(result)
                
                if j % 10 == 0:
                    print(f"   Zoom config {j+1}/{len(zoom_configs)}: {result['metrics']['accuracy']:.3f}")
                    
            except Exception as e:
                print(f"   ❌ Error in zoom config {j+1}: {e}")
                continue
    
    # Combine with original results and sort
    all_results = promising_results + zoom_results
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    
    print(f"\n🏆 FINAL TOP CONFIGURATIONS (including zoom):")
    for i, result in enumerate(all_results[:10], 1):
        cfg = result['config']
        metrics = result['metrics']
        print(f"{i}. {cfg['fusion_strategy']} → Accuracy: {metrics['accuracy']:.3f}")
    
    return all_results

# ----------------------------------------------------------------------- 
# MAIN OPTIMIZATION PIPELINE
# -----------------------------------------------------------------------

def update_live_results(all_results: List[Dict], phase: str, current_config: Dict = None, status: str = "running"):
    """Update a single live results file with current progress"""
    timestamp = int(time.time())
    
    # Create results summary
    live_results = {
        "last_updated": timestamp,
        "status": status,
        "phase": phase,
        "current_config": current_config,
        "total_results": len(all_results),
        "results": all_results
    }
    
    # Save to live results file
    filename = "optimization_live_results.json"
    with open(filename, 'w') as f:
        json.dump(live_results, f, indent=2)
    
    print(f"💾 Updated live results: {filename}")
    return filename

def save_results(results: List[Dict], phase: str):
    """Save results to file"""
    timestamp = int(time.time())
    filename = f"optimization_results_{phase}_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for result in results:
        serializable_result = result.copy()
        if 'results' in serializable_result:
            # Remove detailed results to save space
            del serializable_result['results']
        serializable_results.append(serializable_result)
    
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"💾 Results saved to {filename}")
    return filename

def run_optimization():
    """Main optimization pipeline"""
    print("🚀 Starting Hierarchical Topic Model Optimization")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize overall live results
    update_live_results([], "overall", status="starting")
    
    # Phase 1: BM25 optimization
    print("\n🔄 Starting Phase 1: BM25 Optimization")
    update_live_results([], "overall", {"phase": "phase1_bm25", "status": "running"}, "running")
    bm25_results = run_bm25_optimization()
    save_results(bm25_results, "phase1_bm25")
    
    # Phase 2: Embedding optimization  
    print("\n🔄 Starting Phase 2: Embedding Optimization")
    update_live_results([], "overall", {"phase": "phase2_semantic", "status": "running"}, "running")
    semantic_results = run_embedding_optimization()
    save_results(semantic_results, "phase2_semantic")
    
    # Phase 3: Hybrid combinations
    print("\n🔄 Starting Phase 3: Hybrid Optimization")
    update_live_results([], "overall", {"phase": "phase3_hybrid", "status": "running"}, "running")
    hybrid_results = run_hybrid_optimization(bm25_results, semantic_results)
    save_results(hybrid_results, "phase3_hybrid")
    
    # Phase 4: Zoom into promising configurations
    print("\n🔄 Starting Phase 4: Zoom Optimization")
    update_live_results([], "overall", {"phase": "phase4_final", "status": "running"}, "running")
    final_results = run_zoom_optimization(hybrid_results)
    save_results(final_results, "phase4_final")
    
    total_time = time.time() - start_time
    print(f"\n✅ Optimization completed in {total_time/3600:.1f} hours")
    
    # Update final live results
    if final_results:
        best_result = final_results[0]
        final_summary = {
            "best_accuracy": best_result['metrics']['accuracy'],
            "best_config": best_result['config'],
            "total_time_hours": total_time/3600
        }
        update_live_results(final_results, "overall", final_summary, "completed")
        
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   Strategy: {best_result['config']['fusion_strategy']}")
        print(f"   Accuracy: {best_result['metrics']['accuracy']:.3f}")
        print(f"   Average Rank: {best_result['metrics']['avg_rank']:.2f}")
        print(f"   Top-3 Accuracy: {best_result['metrics']['top3_accuracy']:.3f}")
    else:
        update_live_results([], "overall", {"error": "No results generated"}, "error")

if __name__ == "__main__":
    run_optimization()