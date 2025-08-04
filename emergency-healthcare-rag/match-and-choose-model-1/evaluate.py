#!/usr/bin/env python3
"""
Evaluation script for match-and-choose-model-1
Threshold-based decision making between topic model and LLM choice
"""

import sys
import json
import time
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple, Union
from tqdm import tqdm

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from model import predict, predict_with_details
from search import bm25_search, get_top_topics_with_scores
from config import set_threshold, set_llm_model, get_config_summary

def evaluate_search_only():
    """Evaluate only the search component (BM25) - optimized version"""
    print("🔍 EVALUATING BM25 SEARCH COMPONENT")
    print("=" * 60)
    
    # Load all statements
    statements_dir = Path("data/train/statements")
    answers_dir = Path("data/train/answers")
    
    statements = []
    for path in sorted(statements_dir.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((answers_dir / f"statement_{sid}.json").read_text())
        statements.append((statement, ans["statement_topic"]))
    
    print(f"Testing on {len(statements)} statements")
    
    # Do one search per statement and check all top-k values
    search_results = {}
    top_k_values = range(1, 11)
    
    # Initialize counters for each top-k
    for top_k in top_k_values:
        search_results[top_k] = 0
    
    for stmt, true_topic in tqdm(statements, desc="Search evaluation"):
        # Do one search with top-10 to get all results
        results = bm25_search(stmt, top_k=10)
        
        # Check each top-k value
        for top_k in top_k_values:
            if any(result['topic_id'] == true_topic for result in results[:top_k]):
                search_results[top_k] += 1
    
    # Print results
    for top_k in top_k_values:
        correct = search_results[top_k]
        accuracy = correct / len(statements)
        print(f"Top-{top_k:2d}: {accuracy:.3f} ({correct}/{len(statements)})")
    
    print("-" * 60)
    return search_results

def evaluate_threshold_analysis(threshold: Union[float, str]):
    """Analyze gap distribution and threshold behavior on full dataset"""
    print(f"📊 THRESHOLD ANALYSIS (threshold={threshold})")
    print("=" * 60)
    
    # Load all statements
    statements_dir = Path("data/train/statements")
    answers_dir = Path("data/train/answers")
    
    gap_data = []
    approach_counts = {"separated": 0, "combined": 0}
    
    for path in sorted(statements_dir.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((answers_dir / f"statement_{sid}.json").read_text())
        
        # Get top topics to calculate gap
        top_topics = get_top_topics_with_scores(statement, top_k=5)
        
        if len(top_topics) >= 2:
            gap = top_topics[0]['score'] - top_topics[1]['score']
            expected_topic = ans["statement_topic"]
            
            # Determine which approach would be used
            if threshold == 'NA':
                approach = "separated"
            elif gap > threshold:
                approach = "separated"
            else:
                approach = "combined"
            
            approach_counts[approach] += 1
            
            # Check if expected topic is in top picks
            expected_rank = -1
            for rank, topic_info in enumerate(top_topics, 1):
                if topic_info['topic_id'] == expected_topic:
                    expected_rank = rank
                    break
            
            gap_data.append({
                'gap': gap,
                'approach': approach,
                'expected_rank': expected_rank,
                'is_first_correct': expected_rank == 1
            })
    
    # Analyze results by approach
    separated_cases = [d for d in gap_data if d['approach'] == 'separated']
    combined_cases = [d for d in gap_data if d['approach'] == 'combined']
    
    print(f"Total statements analyzed: {len(gap_data)}")
    print(f"Separated approach: {len(separated_cases)} ({len(separated_cases)/len(gap_data)*100:.1f}%)")
    print(f"Combined approach: {len(combined_cases)} ({len(combined_cases)/len(gap_data)*100:.1f}%)")
    
    if separated_cases:
        sep_accuracy = sum(1 for d in separated_cases if d['is_first_correct']) / len(separated_cases) * 100
        print(f"Separated accuracy (topic model 1st pick): {sep_accuracy:.1f}%")
    
    if combined_cases:
        comb_accuracy = sum(1 for d in combined_cases if d['is_first_correct']) / len(combined_cases) * 100
        print(f"Combined baseline (topic model 1st pick): {comb_accuracy:.1f}%")
        print(f"  (LLM should improve on this baseline)")
    
    print("-" * 60)
    return {
        'approach_counts': approach_counts,
        'separated_accuracy': sep_accuracy if separated_cases else 0,
        'combined_baseline': comb_accuracy if combined_cases else 0
    }

def load_train_data(n_samples: int = 20) -> List[Dict]:
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
                'expected_truth': answer_data['statement_is_true'],
                'expected_topic': answer_data['statement_topic'],
                'sample_id': int(statement_num)
            })
    
    return samples

def evaluate_sample(sample: Dict, sample_num: int, total_samples: int, threshold: Union[float, str]) -> Dict:
    """Evaluate a single sample with detailed decision tracking"""
    statement = sample['statement']
    expected_truth = sample['expected_truth']
    expected_topic = sample['expected_topic']
    
    print(f"Sample {sample_num}/{total_samples}: {statement[:50]}...")
    
    # Time the prediction with detailed info
    start_time = time.time()
    result = predict_with_details(statement, threshold)
    end_time = time.time()
    
    predicted_truth = result['prediction']['truth_value']
    predicted_topic = result['prediction']['topic_id']
    
    # Calculate accuracy
    truth_correct = predicted_truth == expected_truth
    topic_correct = predicted_topic == expected_topic
    
    # Print results with approach info
    truth_symbol = "✅" if truth_correct else "❌"
    topic_symbol = "✅" if topic_correct else "❌"
    approach_symbol = "🔧" if result['decision_info']['approach_used'] == "separated" else "🤖"
    
    print(f"  {truth_symbol} Truth: {predicted_truth} (expected {expected_truth})")
    print(f"  {topic_symbol} Topic: {predicted_topic} (expected {expected_topic})")
    print(f"  {approach_symbol} Approach: {result['decision_info']['approach_used']} (gap: {result['decision_info']['gap']:.3f})")
    print(f"  ⏱️  Time: {end_time - start_time:.2f}s")
    print()
    
    return {
        'sample_id': sample['sample_id'],
        'statement': statement,
        'expected_truth': expected_truth,
        'expected_topic': expected_topic,
        'predicted_truth': predicted_truth,
        'predicted_topic': predicted_topic,
        'truth_correct': truth_correct,
        'topic_correct': topic_correct,
        'time_taken': end_time - start_time,
        'approach_used': result['decision_info']['approach_used'],
        'gap': result['decision_info']['gap'],
        'threshold': result['decision_info']['threshold'],
        'top_topics': result['decision_info']['top_topics']
    }

def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate match-and-choose-model-1')
    parser.add_argument('--samples', type=int, default=20, 
                       help='Number of samples to evaluate (default: 20)')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model to use (e.g., gemma3:27b, llama3.1:8b)')
    parser.add_argument('--threshold', type=str, default=None,
                       help='Threshold to use (float or "NA", default: config default)')
    parser.add_argument('--search-only', action='store_true', 
                       help='Evaluate only search component')
    parser.add_argument('--threshold-analysis', action='store_true', 
                       help='Analyze threshold behavior on full dataset')
    parser.add_argument('--full-pipeline', action='store_true', 
                       help='Evaluate only full pipeline')
    args = parser.parse_args()
    
    # Set model if specified
    if args.model:
        set_llm_model(args.model)
        print(f"🔧 Using specified model: {args.model}")
    
    # Set threshold if specified
    if args.threshold:
        if args.threshold.upper() == 'NA':
            threshold = 'NA'
        else:
            try:
                threshold = float(args.threshold)
            except ValueError:
                print(f"❌ Invalid threshold: {args.threshold}")
                return
        set_threshold(threshold)
        print(f"🎯 Using specified threshold: {threshold}")
    else:
        threshold = None  # Use config default
    
    # Get current configuration
    config = get_config_summary()
    print(f"🔧 Configuration: Model={config['model_info']['name']}, Threshold={config['threshold']}")
    
    # Determine what to evaluate
    if args.search_only:
        # Evaluate search component only
        print("🚑 EMERGENCY HEALTHCARE RAG - MATCH-AND-CHOOSE MODEL 1")
        print("=" * 80)
        print("BM25-only search with threshold-based decision making")
        print("=" * 80)
        search_results = evaluate_search_only()
        print("\n" + "=" * 80)
        print("✅ SEARCH EVALUATION COMPLETE")
        print("=" * 80)
        return
    elif args.threshold_analysis:
        # Analyze threshold behavior
        print("🚑 EMERGENCY HEALTHCARE RAG - MATCH-AND-CHOOSE MODEL 1")
        print("=" * 80)
        print("Threshold Analysis - Decision Behavior on Full Dataset")
        print("=" * 80)
        threshold_results = evaluate_threshold_analysis(config['threshold'])
        print("\n" + "=" * 80)
        print("✅ THRESHOLD ANALYSIS COMPLETE")
        print("=" * 80)
        return
    elif args.full_pipeline:
        # Evaluate full pipeline only
        print("📚 Loading train data...")
        samples = load_train_data(n_samples=args.samples)
        
        if not samples:
            print("❌ No samples loaded. Check if train data exists.")
            return
        
        print(f"📚 Loaded {len(samples)} samples from train data\n")
        print(f"🧪 Evaluating Match-and-Choose Model 1 on {args.samples} samples...\n")
    else:
        # Evaluate both search and full pipeline (default)
        print("🚑 EMERGENCY HEALTHCARE RAG - MATCH-AND-CHOOSE MODEL 1")
        print("=" * 80)
        print("BM25-only search with threshold-based decision making")
        print(f"Threshold: {config['threshold']}, Model: {config['model_info']['name']}")
        print("=" * 80)
        
        # First evaluate search
        search_results = evaluate_search_only()
        print("\n")
        
        # Then threshold analysis
        threshold_results = evaluate_threshold_analysis(config['threshold'])
        print("\n")
        
        # Then evaluate full pipeline
        print("📚 Loading train data...")
        samples = load_train_data(n_samples=args.samples)
        
        if not samples:
            print("❌ No samples loaded. Check if train data exists.")
            return
        
        print(f"📚 Loaded {len(samples)} samples from train data\n")
        print(f"🧪 Evaluating Match-and-Choose Model 1 Full Pipeline on {args.samples} samples...\n")
    
    results = []
    total_time = 0
    approach_counts = {"separated": 0, "combined": 0}
    
    for i, sample in enumerate(samples, 1):
        result = evaluate_sample(sample, i, len(samples), threshold)
        results.append(result)
        total_time += result['time_taken']
        approach_counts[result['approach_used']] += 1
    
    # Calculate summary statistics
    truth_correct = sum(1 for r in results if r['truth_correct'])
    topic_correct = sum(1 for r in results if r['topic_correct'])
    total_correct = truth_correct + topic_correct
    
    truth_accuracy = (truth_correct / len(results)) * 100
    topic_accuracy = (topic_correct / len(results)) * 100
    overall_accuracy = (total_correct / (len(results) * 2)) * 100
    avg_time = total_time / len(results)
    
    # Approach-specific statistics
    separated_results = [r for r in results if r['approach_used'] == 'separated']
    combined_results = [r for r in results if r['approach_used'] == 'combined']
    
    # Print summary
    print("=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Samples: {len(results)}")
    print(f"Truth Accuracy: {truth_accuracy:.1f}% ({truth_correct}/{len(results)})")
    print(f"Topic Accuracy: {topic_accuracy:.1f}% ({topic_correct}/{len(results)})")
    print(f"Overall Accuracy: {overall_accuracy:.1f}% ({total_correct}/{len(results) * 2})")
    print(f"Average Time per Sample: {avg_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Score / {len(results) * 2}: {total_correct}/{len(results) * 2}")
    print()
    print("🎯 APPROACH BREAKDOWN:")
    print(f"Separated approach: {approach_counts['separated']}/{len(results)} ({approach_counts['separated']/len(results)*100:.1f}%)")
    print(f"Combined approach: {approach_counts['combined']}/{len(results)} ({approach_counts['combined']/len(results)*100:.1f}%)")
    
    if separated_results:
        sep_correct = sum(1 for r in separated_results if r['truth_correct'] and r['topic_correct'])
        sep_accuracy = (sep_correct / len(separated_results)) * 100 * 2  # *2 because counting both truth and topic
        print(f"Separated accuracy: {sep_accuracy:.1f}%")
    
    if combined_results:
        comb_correct = sum(1 for r in combined_results if r['truth_correct'] and r['topic_correct'])
        comb_accuracy = (comb_correct / len(combined_results)) * 100 * 2  # *2 because counting both truth and topic
        print(f"Combined accuracy: {comb_accuracy:.1f}%")
    
    print("=" * 80)
    
    # Save detailed results
    output_file = "match-and-choose-model-1/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config': config,
            'summary': {
                'total_samples': len(results),
                'truth_accuracy': truth_accuracy,
                'topic_accuracy': topic_accuracy,
                'overall_accuracy': overall_accuracy,
                'avg_time_per_sample': avg_time,
                'total_time': total_time,
                'score': f"{total_correct}/{len(results) * 2}",
                'approach_counts': approach_counts
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to {output_file}")

if __name__ == "__main__":
    main()