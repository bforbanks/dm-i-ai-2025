#!/usr/bin/env python3
"""
Evaluation script for separated-models-1
Measures accuracy and timing for separated topic/truth classification
"""

import sys
import json
import time
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Tuple

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from model import predict

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

def evaluate_sample(sample: Dict, sample_num: int, total_samples: int) -> Dict:
    """Evaluate a single sample"""
    statement = sample['statement']
    expected_truth = sample['expected_truth']
    expected_topic = sample['expected_topic']
    
    print(f"Sample {sample_num}/{total_samples}: {statement[:50]}...")
    
    # Time the prediction
    start_time = time.time()
    predicted_truth, predicted_topic = predict(statement)
    end_time = time.time()
    
    # Calculate accuracy
    truth_correct = predicted_truth == expected_truth
    topic_correct = predicted_topic == expected_topic
    
    # Print results
    truth_symbol = "✅" if truth_correct else "❌"
    topic_symbol = "✅" if topic_correct else "❌"
    
    print(f"  {truth_symbol} Truth: {predicted_truth} (expected {expected_truth})")
    print(f"  {topic_symbol} Topic: {predicted_topic} (expected {expected_topic})")
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
        'time_taken': end_time - start_time
    }

def main():
    """Main evaluation function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate separated-models-1')
    parser.add_argument('--samples', type=int, default=20, 
                       help='Number of samples to evaluate (default: 20)')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM model to use (e.g., gemma3:27b, llama3.1:8b)')
    args = parser.parse_args()
    
    # Set model if specified
    if args.model:
        os.environ['LLM_MODEL'] = args.model
        print(f"🔧 Using specified model: {args.model}")
    else:
        print(f"🔧 Using default model: {os.getenv('LLM_MODEL', 'gemma3n:e4b')}")
    
    print("📚 Loading train data...")
    samples = load_train_data(n_samples=args.samples)
    
    if not samples:
        print("❌ No samples loaded. Check if train data exists.")
        return
    
    print(f"📚 Loaded {len(samples)} samples from train data\n")
    
    print(f"🧪 Evaluating Separated Models 1 on {args.samples} samples...\n")
    
    results = []
    total_time = 0
    
    for i, sample in enumerate(samples, 1):
        result = evaluate_sample(sample, i, len(samples))
        results.append(result)
        total_time += result['time_taken']
    
    # Calculate summary statistics
    truth_correct = sum(1 for r in results if r['truth_correct'])
    topic_correct = sum(1 for r in results if r['topic_correct'])
    total_correct = truth_correct + topic_correct
    
    truth_accuracy = (truth_correct / len(results)) * 100
    topic_accuracy = (topic_correct / len(results)) * 100
    overall_accuracy = (total_correct / (len(results) * 2)) * 100
    avg_time = total_time / len(results)
    
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
    print("=" * 80)
    
    # Save detailed results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_samples': len(results),
                'truth_accuracy': truth_accuracy,
                'topic_accuracy': topic_accuracy,
                'overall_accuracy': overall_accuracy,
                'avg_time_per_sample': avg_time,
                'total_time': total_time,
                'score': f"{total_correct}/{len(results) * 2}"
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to {output_file}")

if __name__ == "__main__":
    main() 