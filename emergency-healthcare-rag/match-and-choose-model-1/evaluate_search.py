#!/usr/bin/env python3
"""
Enhanced search evaluation for match-and-choose-model-1
Provides detailed topic matching analysis with scores and rankings
"""

import sys
import json
import time
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from search import get_top_topics_with_scores

def load_train_data(n_samples: int = 200) -> List[Dict]:
    """Load first n samples from train data"""
    statements_dir = Path("data/train/statements")
    answers_dir = Path("data/train/answers")
    
    # Get all available statement files
    statement_files = sorted(list(statements_dir.glob("statement_*.txt")))
    
    samples = []
    for i, statement_file in enumerate(statement_files[:n_samples]):
        # Extract the statement number from filename
        statement_num = statement_file.stem.split('_')[1]
        answer_file = answers_dir / f"statement_{statement_num}.json"
        
        if answer_file.exists():
            with open(statement_file, 'r') as f:
                statement = f.read().strip()
            
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
            
            samples.append({
                'statement': statement,
                'expected_topic': answer_data['statement_topic'],
                'sample_id': int(statement_num)
            })
    
    return samples

def evaluate_topic_search(sample: Dict, sample_num: int, total_samples: int) -> Dict:
    """Evaluate topic search for a single sample"""
    statement = sample['statement']
    expected_topic = sample['expected_topic']
    
    print(f"Sample {sample_num}/{total_samples}: {statement[:50]}...")
    
    # Time the search
    start_time = time.time()
    top_topics = get_top_topics_with_scores(statement, top_k=10)
    end_time = time.time()
    
    # Find where the expected topic appears in the results
    expected_topic_found = False
    expected_topic_rank = -1
    expected_topic_score = 0.0
    
    # Check if expected topic is in top 5
    for rank, topic_info in enumerate(top_topics[:5], 1):
        if topic_info['topic_id'] == expected_topic:
            expected_topic_found = True
            expected_topic_rank = rank
            expected_topic_score = topic_info['score']
            break
    
    # If expected topic not in top 5, add it to the results for analysis
    if not expected_topic_found:
        # Get all topics to find the expected one
        all_topics = get_top_topics_with_scores(statement, top_k=50)
        for rank, topic_info in enumerate(all_topics, 1):
            if topic_info['topic_id'] == expected_topic:
                expected_topic_rank = rank
                expected_topic_score = topic_info['score']
                break
    
    # Print results
    rank_symbol = "✅" if expected_topic_rank <= 5 else "❌"
    print(f"  {rank_symbol} Expected topic {expected_topic} found at rank {expected_topic_rank} (score: {expected_topic_score:.3f})")
    print(f"  ⏱️  Time: {end_time - start_time:.3f}s")
    
    # Show top 5 results
    print("  Top 5 matches:")
    for rank, topic_info in enumerate(top_topics[:5], 1):
        is_expected = topic_info['topic_id'] == expected_topic
        marker = "★" if is_expected else " "
        print(f"    {marker} {rank}. Topic {topic_info['topic_id']} ({topic_info['topic_name']}) - Score: {topic_info['score']:.3f}")
    
    print()
    
    return {
        'sample_id': sample['sample_id'],
        'statement': statement,
        'expected_topic': expected_topic,
        'expected_topic_rank': expected_topic_rank,
        'expected_topic_score': expected_topic_score,
        'top_topics': top_topics[:10],  # Store top 10 for analysis
        'time_taken': end_time - start_time
    }

def analyze_results(results: List[Dict]) -> Dict:
    """Analyze the search results and provide detailed statistics"""
    total_samples = len(results)
    
    # Initialize counters for each rank
    rank_counts = {i: 0 for i in range(1, 11)}  # Top 1-10
    rank_scores = {i: [] for i in range(1, 11)}  # Scores for each rank
    
    # Count occurrences and collect scores
    for result in results:
        rank = result['expected_topic_rank']
        score = result['expected_topic_score']
        
        if 1 <= rank <= 10:
            rank_counts[rank] += 1
            rank_scores[rank].append(score)
    
    # Calculate percentages and average scores
    analysis = {}
    cumulative_correct = 0
    
    for rank in range(1, 11):
        count = rank_counts[rank]
        percentage = (count / total_samples) * 100
        cumulative_correct += count
        cumulative_percentage = (cumulative_correct / total_samples) * 100
        
        avg_score = sum(rank_scores[rank]) / len(rank_scores[rank]) if rank_scores[rank] else 0.0
        
        analysis[f'rank_{rank}'] = {
            'count': count,
            'percentage': percentage,
            'cumulative_count': cumulative_correct,
            'cumulative_percentage': cumulative_percentage,
            'avg_score': avg_score
        }
    
    # Calculate overall statistics
    total_time = sum(r['time_taken'] for r in results)
    avg_time = total_time / total_samples
    
    analysis['overall'] = {
        'total_samples': total_samples,
        'avg_time_per_search': avg_time,
        'total_time': total_time
    }
    
    return analysis

def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate topic search for match-and-choose-model-1')
    parser.add_argument('--samples', type=int, default=200, 
                       help='Number of samples to evaluate (default: 200)')
    args = parser.parse_args()
    
    print("🔍 TOPIC SEARCH EVALUATION - MATCH-AND-CHOOSE-MODEL-1")
    print("=" * 80)
    print("BM25-only search with optimized chunking (chunk_size=128, overlap=12)")
    print("=" * 80)
    
    print(f"📚 Loading train data...")
    samples = load_train_data(n_samples=args.samples)
    
    if not samples:
        print("❌ No samples loaded. Check if train data exists.")
        return
    
    print(f"📚 Loaded {len(samples)} samples from train data\n")
    print(f"🧪 Evaluating topic search on {args.samples} samples...\n")
    
    results = []
    
    for i, sample in enumerate(samples, 1):
        result = evaluate_topic_search(sample, i, len(samples))
        results.append(result)
    
    # Analyze results
    analysis = analyze_results(results)
    
    # Print detailed analysis
    print("=" * 80)
    print("📊 TOPIC SEARCH ANALYSIS")
    print("=" * 80)
    print(f"Total Samples: {analysis['overall']['total_samples']}")
    print(f"Average Time per Search: {analysis['overall']['avg_time_per_search']:.3f}s")
    print(f"Total Time: {analysis['overall']['total_time']:.2f}s")
    print()
    
    print("Rank Analysis:")
    print("-" * 60)
    print("Rank | Count | % | Cum.% | Avg Score")
    print("-" * 60)
    
    for rank in range(1, 11):
        rank_data = analysis[f'rank_{rank}']
        print(f"{rank:4d} | {rank_data['count']:5d} | {rank_data['percentage']:5.1f}% | {rank_data['cumulative_percentage']:5.1f}% | {rank_data['avg_score']:.3f}")
    
    print("-" * 60)
    
    # Save detailed results
    output_file = "match-and-choose-model-1/search_evaluation_results.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'analysis': analysis,
            'detailed_results': [
                {
                    'sample_id': r['sample_id'],
                    'expected_topic': r['expected_topic'],
                    'expected_topic_rank': r['expected_topic_rank'],
                    'expected_topic_score': r['expected_topic_score'],
                    'time_taken': r['time_taken']
                }
                for r in results
            ]
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to {output_file}")
    print("=" * 80)

if __name__ == "__main__":
    main() 