#!/usr/bin/env python3
"""
🚀 STREAMLINED: Topic Search Optimization
Finds optimal search configuration with automatic result saving.

FOCUSED: Core optimization without over-engineering
SAVES: Results after each configuration evaluation
"""

import json
import time
import pickle
import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add current directory for imports
sys.path.insert(0, str(Path(__file__).parent))

def check_setup():
    """Quick setup check"""
    print("🔧 Setup Check...")
    
    try:
        import numpy
        from rank_bm25 import BM25Okapi
        print("✅ Core packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: pip install rank-bm25 numpy")
        sys.exit(1)
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers available")
        return True
    except ImportError:
        print("⚠️  sentence-transformers not available - BM25-only mode")
        return False

SENTENCE_TRANSFORMERS_AVAILABLE = check_setup()

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip install tqdm")
    from tqdm import tqdm

if SENTENCE_TRANSFORMERS_AVAILABLE:
    from sentence_transformers import SentenceTransformer

from rank_bm25 import BM25Okapi

class SearchOptimizer:
    """Streamlined search optimizer with automatic result saving"""
    
    def __init__(self):
        self.results = []
        self.results_file = Path("match-and-choose-model-1/optimization_results.json")
        
    def load_topics_mapping(self) -> Dict[str, int]:
        """Load topic name to ID mapping"""
        with open('data/topics.json', 'r') as f:
            return json.load(f)
    
    def chunk_words(self, words: List[str], size: int, overlap: int) -> List[str]:
        """Create word chunks with specified size and overlap"""
        step = max(1, size - overlap)
        chunks = []
        for i in range(0, len(words), step):
            chunk = words[i : i + size]
            if len(chunk) < 10:
                continue
            chunks.append(" ".join(chunk))
            if i + size >= len(words):
                break
        return chunks
    
    def build_search_index(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build search index based on configuration"""
        use_condensed = config.get('use_condensed_topics', True)
        topics_type = "condensed" if use_condensed else "regular"
        cache_key = f"{config['search_type']}_{config['chunk_size']}_{config['overlap']}_{topics_type}"
        
        print(f"🔨 Building index: {cache_key}")
        
        chunks = []
        topics = []
        chunk_texts = []
        topic_names = []
        embeddings = None
        
        topics_data = self.load_topics_mapping()
        
        if use_condensed:
            topic_dir = Path("data/condensed_topics")
        else:
            topic_dir = Path("data/topics")
        
        if not topic_dir.exists():
            raise FileNotFoundError(f"Topics directory not found: {topic_dir}")
        
        for md_file in topic_dir.rglob("*.md"):
            topic_name = md_file.parent.name
            topic_id = topics_data.get(topic_name, -1)
            
            if topic_id == -1:
                continue
                
            words = md_file.read_text(encoding="utf-8").split()
            for chunk_text in self.chunk_words(words, config['chunk_size'], config['overlap']):
                chunks.append(chunk_text)
                topics.append(topic_id)
                chunk_texts.append(chunk_text)
                topic_names.append(topic_name)
        
        # Build BM25 index
        tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]
        bm25 = BM25Okapi(tokenized_chunks)
        
        # Build embeddings if hybrid search
        if config['search_type'] == 'hybrid' and SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"🧠 Computing embeddings with {config.get('embedding', 'all-mpnet-base-v2')}")
            model = SentenceTransformer(config.get('embedding', 'all-mpnet-base-v2'))
            embeddings = model.encode(chunk_texts, show_progress_bar=True)
        
        return {
            'topics': topics,
            'chunk_texts': chunk_texts,
            'topic_names': topic_names,
            'bm25': bm25,
            'embeddings': embeddings,
            'topics_data': topics_data,
            'config': config
        }
    
    def bm25_search(self, data: Dict, statement: str, top_k: int = 10) -> List[Dict]:
        """Perform BM25 search"""
        tokenized_query = statement.lower().split()
        scores = data['bm25'].get_scores(tokenized_query)
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'topic_id': data['topics'][idx],
                'topic_name': data['topic_names'][idx],
                'chunk_text': data['chunk_texts'][idx],
                'score': float(scores[idx])
            })
        
        return results
    
    def hybrid_search(self, data: Dict, statement: str, config: Dict, top_k: int = 10) -> List[Dict]:
        """Perform hybrid BM25 + semantic search with RRF"""
        if data['embeddings'] is None:
            return self.bm25_search(data, statement, top_k)
        
        # BM25 search
        tokenized_query = statement.lower().split()
        bm25_scores = data['bm25'].get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # Semantic search
        model = SentenceTransformer(config.get('embedding', 'all-mpnet-base-v2'))
        query_embedding = model.encode([statement])
        similarities = np.dot(data['embeddings'], query_embedding.T).flatten()
        vector_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        rrf_k = config.get('rrf_k', 60)
        
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + rrf_k)
        
        for rank, idx in enumerate(vector_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + rrf_k)
        
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_indices[:top_k]:
            results.append({
                'topic_id': data['topics'][idx],
                'topic_name': data['topic_names'][idx],
                'chunk_text': data['chunk_texts'][idx],
                'score': float(score)
            })
        
        return results
    
    def get_top_topics_with_scores(self, data: Dict, statement: str, config: Dict, top_k: int = 10) -> List[Dict]:
        """Get top topics with deduplication by topic_id"""
        if config['search_type'] == 'hybrid':
            search_results = self.hybrid_search(data, statement, config, top_k * 3)
        else:
            search_results = self.bm25_search(data, statement, top_k * 3)
        
        # Group by topic_id and keep best score
        topic_scores = {}
        for result in search_results:
            topic_id = result['topic_id']
            score = result['score']
            
            if topic_id not in topic_scores or score > topic_scores[topic_id]['score']:
                topic_scores[topic_id] = {
                    'topic_id': topic_id,
                    'topic_name': result['topic_name'],
                    'score': score,
                    'best_chunk': result['chunk_text']
                }
        
        sorted_topics = sorted(topic_scores.values(), key=lambda x: x['score'], reverse=True)
        return sorted_topics[:top_k]
    
    def evaluate_search_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single search configuration"""
        print(f"\n🔍 Evaluating: {config['name']}")
        
        start_time = time.time()
        data = self.build_search_index(config)
        build_time = time.time() - start_time
        
        # Load test statements
        statements_dir = Path("data/train/statements")
        answers_dir = Path("data/train/answers")
        
        statements = []
        for path in sorted(statements_dir.glob("*.txt")):
            sid = path.stem.split("_")[1]
            statement = path.read_text().strip()
            answer_path = answers_dir / f"statement_{sid}.json"
            
            if answer_path.exists():
                answer = json.loads(answer_path.read_text())
                statements.append((statement, answer["statement_topic"]))
        
        print(f"📊 Testing on {len(statements)} statements")
        
        # Evaluate search accuracy
        results = {k: 0 for k in [1, 3, 5, 10]}
        search_times = []
        
        for stmt, true_topic in tqdm(statements, desc="Evaluating"):
            search_start = time.time()
            top_topics = self.get_top_topics_with_scores(data, stmt, config, top_k=10)
            search_time = time.time() - search_start
            search_times.append(search_time)
            
            found_topics = [topic['topic_id'] for topic in top_topics]
            for k in results.keys():
                if true_topic in found_topics[:k]:
                    results[k] += 1
        
        total_statements = len(statements)
        accuracy_metrics = {}
        for k, correct in results.items():
            accuracy_metrics[f'top_{k}_accuracy'] = correct / total_statements
        
        result = {
            'config': config,
            'accuracy_metrics': accuracy_metrics,
            'avg_search_time': np.mean(search_times),
            'total_build_time': build_time,
            'total_statements': total_statements,
            'detailed_results': results
        }
        
        # Save result immediately
        self.save_result(result)
        
        return result
    
    def save_result(self, result: Dict[str, Any]):
        """Save a single result to file"""
        self.results.append(result)
        
        output_data = {
            'results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_configs': len(self.results)
        }
        
        with open(self.results_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Result saved: {result['config']['name']} - Top-1: {result['accuracy_metrics']['top_1_accuracy']*100:.1f}%")
    
    def run_optimization(self) -> List[Dict[str, Any]]:
        """Run streamlined optimization with focused configurations"""
        
        # Load existing results if available
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                existing_data = json.load(f)
                self.results = existing_data.get('results', [])
                print(f"📋 Loaded {len(self.results)} existing results")
        
        # Focused configurations - only the most important ones
        configs = [
            # Baseline BM25
            {"search_type": "bm25", "chunk_size": 128, "overlap": 12, "use_condensed_topics": True, "name": "baseline_bm25"},
            
            # BM25 variations
            {"search_type": "bm25", "chunk_size": 96, "overlap": 8, "use_condensed_topics": True, "name": "bm25_96_8"},
            {"search_type": "bm25", "chunk_size": 192, "overlap": 16, "use_condensed_topics": True, "name": "bm25_192_16"},
            
            # Test regular topics
            {"search_type": "bm25", "chunk_size": 128, "overlap": 12, "use_condensed_topics": False, "name": "bm25_regular"},
        ]
        
        # Add hybrid configs if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            configs.extend([
                {"search_type": "hybrid", "chunk_size": 256, "overlap": 24, "rrf_k": 60, 
                 "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_256_24"},
                {"search_type": "hybrid", "chunk_size": 384, "overlap": 32, "rrf_k": 60, 
                 "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_384_32"},
                {"search_type": "hybrid", "chunk_size": 256, "overlap": 24, "rrf_k": 30, 
                 "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_rrf_30"},
            ])
        
        # Skip already evaluated configs
        evaluated_names = {r['config']['name'] for r in self.results}
        remaining_configs = [c for c in configs if c['name'] not in evaluated_names]
        
        if not remaining_configs:
            print("✅ All configurations already evaluated!")
            return self.results
        
        print(f"🚀 Evaluating {len(remaining_configs)} remaining configurations...")
        
        for i, config in enumerate(remaining_configs, 1):
            print(f"\n📊 Configuration {i}/{len(remaining_configs)}: {config['name']}")
            
            try:
                result = self.evaluate_search_config(config)
                print(f"✅ Top-1 Accuracy: {result['accuracy_metrics']['top_1_accuracy']*100:.1f}%")
                print(f"⏱️  Avg Search Time: {result['avg_search_time']*1000:.1f}ms")
                
            except Exception as e:
                print(f"❌ Configuration failed: {e}")
                error_result = {
                    'config': config,
                    'error': str(e),
                    'accuracy_metrics': {'top_1_accuracy': 0}
                }
                self.save_result(error_result)
        
        return self.results
    
    def print_summary(self):
        """Print summary of results"""
        if not self.results:
            print("❌ No results to display")
            return
        
        print("\n📊 OPTIMIZATION RESULTS SUMMARY")
        print("=" * 80)
        print(f"{'Config':<20} {'Type':<8} {'Topics':<10} {'Top-1':<8} {'Top-3':<8} {'Time':<8}")
        print("-" * 80)
        
        for result in self.results:
            if 'error' in result:
                config = result['config']
                print(f"{config.get('name', 'unknown'):<20} {'ERROR':<8} {'-':<10} {'-':<8} {'-':<8} {'-':<8}")
                continue
                
            config = result['config']
            metrics = result['accuracy_metrics']
            topics_type = "condensed" if config.get('use_condensed_topics', True) else "regular"
            
            print(f"{config.get('name', 'unknown'):<20} "
                  f"{config['search_type']:<8} "
                  f"{topics_type:<10} "
                  f"{metrics['top_1_accuracy']*100:6.1f}% "
                  f"{metrics['top_3_accuracy']*100:6.1f}% "
                  f"{result['avg_search_time']*1000:6.1f}ms")
        
        # Find best configuration
        valid_results = [r for r in self.results if 'error' not in r]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['accuracy_metrics']['top_1_accuracy'])
            print(f"\n🏆 BEST CONFIGURATION: {best_result['config']['name']}")
            print(f"🎯 Top-1 Accuracy: {best_result['accuracy_metrics']['top_1_accuracy']*100:.1f}%")
            print(f"⏱️  Avg Search Time: {best_result['avg_search_time']*1000:.1f}ms")

def main():
    """Streamlined optimization pipeline"""
    print("🚀 STREAMLINED SEARCH OPTIMIZATION")
    print("=" * 50)
    
    # Check data files
    required_files = ["data/topics.json", "data/condensed_topics", "data/train"]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"❌ Missing data files: {missing}")
        sys.exit(1)
    
    print("✅ Data files found")
    
    # Run optimization
    optimizer = SearchOptimizer()
    results = optimizer.run_optimization()
    
    # Show summary
    optimizer.print_summary()
    
    print(f"\n💾 Results saved to: {optimizer.results_file}")
    print("🎉 Optimization complete!")

if __name__ == "__main__":
    main()