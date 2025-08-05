#!/usr/bin/env python3
"""
🚀 SINGLE SCRIPT: Topic Search Grid Search Optimization
Finds optimal search configuration to maximize topic matching accuracy.

ALL-IN-ONE: Setup check + optimization + results + implementation guide

Models used: ALL LOCAL (no API calls)
- BM25: Local keyword search 
- Embeddings: sentence-transformers models (downloaded locally)
"""

import json
import time
import pickle
import numpy as np
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add current directory for imports
sys.path.insert(0, str(Path(__file__).parent))

def check_setup():
    """Quick setup check - exit if critical dependencies missing"""
    print("🔧 Setup Check...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    # Check required packages
    try:
        import numpy
        from rank_bm25 import BM25Okapi
        print("✅ Core packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Install with: pip install rank-bm25 numpy")
        sys.exit(1)
    
    # Check optional packages
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers available - hybrid search enabled")
        return True
    except ImportError:
        print("⚠️  sentence-transformers not available - BM25-only mode")
        print("   Install with: pip install sentence-transformers")
        return False

# Run setup check
SENTENCE_TRANSFORMERS_AVAILABLE = check_setup()

# Import after check
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
    """Grid search optimizer for topic search configurations"""
    
    def __init__(self):
        self.results = []
        self.data_cache = {}
        
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
            if len(chunk) < 10:  # Skip tiny fragments
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
        
        if cache_key in self.data_cache:
            print(f"📋 Using cached index: {cache_key}")
            return self.data_cache[cache_key]
        
        print(f"🔨 Building search index: {cache_key}")
        
        chunks = []
        topics = []
        chunk_texts = []
        topic_names = []
        embeddings = None
        
        # Load topics mapping
        topics_data = self.load_topics_mapping()
        
        # Choose topic directory based on configuration
        if use_condensed:
            topic_dir = Path("data/condensed_topics")
            print(f"📁 Using condensed topics from {topic_dir}")
        else:
            topic_dir = Path("data/topics")
            print(f"📁 Using regular topics from {topic_dir}")
        
        if not topic_dir.exists():
            raise FileNotFoundError(f"Topics directory not found: {topic_dir}")
        
        if not any(topic_dir.rglob("*.md")):
            raise FileNotFoundError(f"No .md files found in topics directory: {topic_dir}")
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
        
        data = {
            'topics': topics,
            'chunk_texts': chunk_texts,
            'topic_names': topic_names,
            'bm25': bm25,
            'embeddings': embeddings,
            'topics_data': topics_data,
            'config': config
        }
        
        self.data_cache[cache_key] = data
        return data
    
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
            print("⚠️  No embeddings available, falling back to BM25")
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
        
        # Score BM25 results
        for rank, idx in enumerate(bm25_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + rrf_k)
        
        # Score semantic results
        for rank, idx in enumerate(vector_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + rrf_k)
        
        # Sort by combined scores
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build results
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
        
        # Sort by score and return top-k
        sorted_topics = sorted(topic_scores.values(), key=lambda x: x['score'], reverse=True)
        return sorted_topics[:top_k]
    
    def evaluate_search_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single search configuration"""
        print(f"\n🔍 Evaluating: {config}")
        
        # Build search index
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
        
        # Evaluate search accuracy for different top-k values
        results = {k: 0 for k in [1, 3, 5, 10]}
        search_times = []
        
        for stmt, true_topic in tqdm(statements, desc="Evaluating"):
            search_start = time.time()
            top_topics = self.get_top_topics_with_scores(data, stmt, config, top_k=10)
            search_time = time.time() - search_start
            search_times.append(search_time)
            
            # Check accuracy for different top-k values
            found_topics = [topic['topic_id'] for topic in top_topics]
            for k in results.keys():
                if true_topic in found_topics[:k]:
                    results[k] += 1
        
        # Calculate metrics
        total_statements = len(statements)
        accuracy_metrics = {}
        for k, correct in results.items():
            accuracy_metrics[f'top_{k}_accuracy'] = correct / total_statements
        
        return {
            'config': config,
            'accuracy_metrics': accuracy_metrics,
            'avg_search_time': np.mean(search_times),
            'total_build_time': build_time,
            'total_statements': total_statements,
            'detailed_results': results
        }
    
    def run_grid_search(self) -> List[Dict[str, Any]]:
        """Run the strategic 10-configuration grid search"""
        
        # Strategic grid focused on highest-impact parameters
        base_configs = [
            # Baseline BM25-only (current approach with condensed topics)
            {"search_type": "bm25", "chunk_size": 128, "overlap": 12, "use_condensed_topics": True, "name": "baseline_bm25_condensed"},
            
            # Test regular vs condensed topics
            {"search_type": "bm25", "chunk_size": 128, "overlap": 12, "use_condensed_topics": False, "name": "baseline_bm25_regular"},
            
            # BM25-only parameter optimization (using condensed)
            {"search_type": "bm25", "chunk_size": 96, "overlap": 8, "use_condensed_topics": True, "name": "bm25_96_8_condensed"},
            {"search_type": "bm25", "chunk_size": 192, "overlap": 16, "use_condensed_topics": True, "name": "bm25_192_16_condensed"},
        ]
        
        hybrid_configs = [
            # Hybrid search exploration - different chunk sizes (condensed topics)
            {"search_type": "hybrid", "chunk_size": 128, "overlap": 12, "rrf_k": 60, 
             "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_128_12_condensed"},
            {"search_type": "hybrid", "chunk_size": 256, "overlap": 24, "rrf_k": 60, 
             "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_256_24_condensed"},
            {"search_type": "hybrid", "chunk_size": 384, "overlap": 32, "rrf_k": 60, 
             "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_384_32_condensed"},
            
            # Test hybrid with regular topics
            {"search_type": "hybrid", "chunk_size": 256, "overlap": 24, "rrf_k": 60, 
             "embedding": "all-mpnet-base-v2", "use_condensed_topics": False, "name": "hybrid_256_24_regular"},
            
            # RRF parameter optimization
            {"search_type": "hybrid", "chunk_size": 256, "overlap": 24, "rrf_k": 30, 
             "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_rrf_30"},
            {"search_type": "hybrid", "chunk_size": 256, "overlap": 24, "rrf_k": 90, 
             "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_rrf_90"},
            
            # Different embedding model
            {"search_type": "hybrid", "chunk_size": 256, "overlap": 24, "rrf_k": 60, 
             "embedding": "all-MiniLM-L6-v2", "use_condensed_topics": True, "name": "hybrid_minilm"},
            
            # Best hybrid variant
            {"search_type": "hybrid", "chunk_size": 320, "overlap": 28, "rrf_k": 45, 
             "embedding": "all-mpnet-base-v2", "use_condensed_topics": True, "name": "hybrid_optimized"},
        ]
        
        # Choose configs based on availability
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            grid_configs = base_configs + hybrid_configs
        else:
            grid_configs = base_configs
            print("⚠️  Running BM25-only configs (sentence-transformers not available)")
        
        print("🚀 Starting Strategic Grid Search Optimization")
        print(f"📋 Testing {len(grid_configs)} configurations")
        print("=" * 80)
        
        results = []
        best_accuracy = 0
        best_config = None
        
        for i, config in enumerate(grid_configs, 1):
            print(f"\n📊 Configuration {i}/{len(grid_configs)}: {config['name']}")
            
            try:
                result = self.evaluate_search_config(config)
                results.append(result)
                
                # Track best configuration
                top1_acc = result['accuracy_metrics']['top_1_accuracy']
                if top1_acc > best_accuracy:
                    best_accuracy = top1_acc
                    best_config = config
                
                print(f"✅ Top-1 Accuracy: {top1_acc:.3f} ({top1_acc*100:.1f}%)")
                print(f"⏱️  Avg Search Time: {result['avg_search_time']:.4f}s")
                
            except Exception as e:
                print(f"❌ Configuration failed: {e}")
                results.append({
                    'config': config,
                    'error': str(e),
                    'accuracy_metrics': {'top_1_accuracy': 0}
                })
        
        print("\n" + "=" * 80)
        print("🎯 OPTIMIZATION COMPLETE!")
        print(f"🏆 Best Configuration: {best_config['name'] if best_config else 'None'}")
        print(f"🎯 Best Top-1 Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "search_optimization_results.json"):
        """Save optimization results to file"""
        output_data = {
            'optimization_summary': {
                'total_configs': len(results),
                'completed_configs': len([r for r in results if 'error' not in r]),
                'best_config': max(results, key=lambda x: x.get('accuracy_metrics', {}).get('top_1_accuracy', 0))
            },
            'detailed_results': results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        output_path = Path("match-and-choose-model-1") / filename
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        return output_path
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary table of results"""
        print("\n📊 OPTIMIZATION RESULTS SUMMARY")
        print("=" * 120)
        print(f"{'Config Name':<25} {'Type':<8} {'Topics':<10} {'Chunk':<6} {'Overlap':<7} {'Top-1':<8} {'Top-3':<8} {'Top-5':<8} {'Time':<8}")
        print("-" * 120)
        
        for result in results:
            if 'error' in result:
                config = result['config']
                print(f"{config.get('name', 'unknown'):<25} {'ERROR':<8} {'-':<10} {'-':<6} {'-':<7} {'-':<8} {'-':<8} {'-':<8} {'-':<8}")
                continue
                
            config = result['config']
            metrics = result['accuracy_metrics']
            topics_type = "condensed" if config.get('use_condensed_topics', True) else "regular"
            
            print(f"{config.get('name', 'unknown'):<25} "
                  f"{config['search_type']:<8} "
                  f"{topics_type:<10} "
                  f"{config['chunk_size']:<6} "
                  f"{config['overlap']:<7} "
                  f"{metrics['top_1_accuracy']*100:6.1f}% "
                  f"{metrics['top_3_accuracy']*100:6.1f}% "
                  f"{metrics['top_5_accuracy']*100:6.1f}% "
                  f"{result['avg_search_time']*1000:6.1f}ms")
        
        print("=" * 120)
    
    def show_best_config_implementation(self, results: List[Dict[str, Any]]):
        """Show best configuration and implementation code"""
        if not results:
            print("❌ No results to analyze")
            return
            
        # Find best configuration
        best_result = max(results, key=lambda x: x.get('accuracy_metrics', {}).get('top_1_accuracy', 0))
        config = best_result['config']
        metrics = best_result['accuracy_metrics']
        
        print("\n" + "🏆 BEST CONFIGURATION FOUND" + "🏆")
        print("=" * 60)
        print(f"🎯 Configuration: {config.get('name', 'unknown')}")
        print(f"🔍 Search Type: {config['search_type']}")
        print(f"📁 Topics: {'condensed' if config.get('use_condensed_topics', True) else 'regular'}")
        print(f"📏 Chunk Size: {config['chunk_size']} words")
        print(f"🔄 Overlap: {config['overlap']} words")
        
        if config['search_type'] == 'hybrid':
            print(f"🧠 Embedding Model: {config.get('embedding', 'all-mpnet-base-v2')}")
            print(f"🔀 RRF K Parameter: {config.get('rrf_k', 60)}")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   Top-1 Accuracy: {metrics['top_1_accuracy']*100:.1f}%")
        print(f"   Top-3 Accuracy: {metrics['top_3_accuracy']*100:.1f}%")
        print(f"   Top-5 Accuracy: {metrics['top_5_accuracy']*100:.1f}%")
        print(f"   Avg Search Time: {best_result['avg_search_time']*1000:.1f}ms")
        
        # Show improvement
        baseline = 0.895  # Current baseline
        improvement = metrics['top_1_accuracy'] - baseline
        if improvement > 0:
            print(f"   🚀 Improvement: +{improvement*100:.1f}% vs baseline ({baseline*100:.1f}%)")
        else:
            print(f"   📊 Change: {improvement*100:+.1f}% vs baseline ({baseline*100:.1f}%)")
        
        print("\n" + "💻 IMPLEMENTATION INSTRUCTIONS" + "💻")
        print("=" * 60)
        
        topics_dir = "data/condensed_topics" if config.get('use_condensed_topics', True) else "data/topics"
        
        if config['search_type'] == 'bm25':
            print(f"""
🔧 UPDATE YOUR search.py FILE:

1. Change chunk parameters:
   CHUNK_SIZE = {config['chunk_size']}
   OVERLAP = {config['overlap']}

2. Update topics directory:
   topic_dir = Path("{topics_dir}")

3. Your search remains BM25-only - just update chunking and topics!

4. Expected improvement: +{improvement*100:.1f}% accuracy
""")
        else:  # hybrid
            print(f"""
🔧 IMPLEMENT HYBRID SEARCH in your search.py:

1. Install dependencies (if not already):
   pip install sentence-transformers

2. Add these parameters:
   CHUNK_SIZE = {config['chunk_size']}
   OVERLAP = {config['overlap']}
   EMBEDDING_MODEL = "{config.get('embedding', 'all-mpnet-base-v2')}"
   RRF_K = {config.get('rrf_k', 60)}

3. Update topics directory:
   topic_dir = Path("{topics_dir}")

4. Add hybrid search function:
   def hybrid_search(statement, data, top_k=10):
       # BM25 search
       tokenized_query = statement.lower().split()
       bm25_scores = data['bm25'].get_scores(tokenized_query)
       bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
       
       # Semantic search (if embeddings available)
       if 'embeddings' in data and data['embeddings'] is not None:
           model = SentenceTransformer(EMBEDDING_MODEL)
           query_embedding = model.encode([statement])
           similarities = np.dot(data['embeddings'], query_embedding.T).flatten()
           vector_indices = np.argsort(similarities)[-top_k:][::-1]
           
           # Reciprocal Rank Fusion
           rrf_scores = {{}}
           for rank, idx in enumerate(bm25_indices):
               rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + RRF_K)
           for rank, idx in enumerate(vector_indices):
               rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + RRF_K)
           
           # Sort and return results
           sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
           return [build_result_from_index(idx, score) for idx, score in sorted_indices[:top_k]]
       else:
           # Fallback to BM25
           return [build_result_from_index(idx, bm25_scores[idx]) for idx in bm25_indices]

5. During index building, compute embeddings:
   embeddings = model.encode(chunk_texts, show_progress_bar=True)

6. Expected improvement: +{improvement*100:.1f}% accuracy
""")
        
        return config

def main():
    """ALL-IN-ONE: Complete optimization pipeline"""
    print("🚀 TOPIC SEARCH OPTIMIZATION - SINGLE SCRIPT")
    print("=" * 60)
    print("🔍 Finding optimal search configuration...")
    print("🏠 All models run locally (no API calls)")
    print("⏱️  Expected runtime: 1-3 hours")
    print("=" * 60)
    
    # Check data files
    required_files = ["data/topics.json", "data/condensed_topics", "data/train"]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"❌ Missing data files: {missing}")
        print("   Make sure you're in the emergency-healthcare-rag/ directory")
        sys.exit(1)
    
    print("✅ Data files found")
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("📊 Running BM25-only optimization (install sentence-transformers for hybrid)")
    else:
        print("🎯 Running full optimization (BM25 + hybrid search)")
    
    # Initialize and run optimization
    optimizer = SearchOptimizer()
    
    print(f"\n🚀 Starting optimization...")
    results = optimizer.run_grid_search()
    
    # Show results
    optimizer.print_summary(results)
    
    # Save detailed results
    output_path = optimizer.save_results(results)
    print(f"\n💾 Detailed results: {output_path}")
    
    # Show best configuration and implementation
    optimizer.show_best_config_implementation(results)
    
    print("\n" + "🎉 OPTIMIZATION COMPLETE!" + "🎉")
    print("=" * 60)
    print("✅ Follow the implementation instructions above")
    print("✅ Test with: python match-and-choose-model-1/evaluate.py")
    print("✅ Expected improvement in topic matching accuracy!")

if __name__ == "__main__":
    main()