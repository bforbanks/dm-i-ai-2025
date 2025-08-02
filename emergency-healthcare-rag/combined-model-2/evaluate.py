#!/usr/bin/env python3
"""
Evaluation script for combined-model-2
Loads data from train directory and measures accuracy and performance
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the main model which should use combined-model-2
from model import predict

def load_train_data(n_samples: int = 10) -> List[Dict]:
    """
    Load the first n samples from train data starting from statement_0000
    """
    statements_dir = Path("data/train/statements")
    answers_dir = Path("data/train/answers")
    
    samples = []
    for i in range(n_samples):
        statement_file = statements_dir / f"statement_{i:04d}.txt"
        answer_file = answers_dir / f"statement_{i:04d}.json"
        
        if not statement_file.exists():
            print(f"Warning: Statement file {statement_file.name} not found, stopping")
            break
            
        if not answer_file.exists():
            print(f"Warning: Answer file {answer_file.name} not found, skipping")
            continue
            
        # Load statement
        with open(statement_file, 'r', encoding='utf-8') as f:
            statement = f.read().strip()
            
        # Load answer
        with open(answer_file, 'r', encoding='utf-8') as f:
            answer = json.load(f)
            
        samples.append({
            'id': f"statement_{i:04d}",
            'statement': statement,
            'expected_truth': answer['statement_is_true'],
            'expected_topic': answer['statement_topic']
        })
    
    return samples

def evaluate_model(samples: List[Dict]) -> Dict:
    """
    Evaluate the model on the given samples
    """
    print(f"🧪 Evaluating Combined Model 2 on {len(samples)} samples...\n")
    
    results = []
    total_time = 0
    correct_truth = 0
    correct_topic = 0
    
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}/{len(samples)}: {sample['statement'][:60]}...")
        
        # Time the prediction
        start_time = time.time()
        try:
            predicted_truth, predicted_topic = predict(sample['statement'])
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Check accuracy
            truth_correct = predicted_truth == sample['expected_truth']
            topic_correct = predicted_topic == sample['expected_topic']
            
            if truth_correct:
                correct_truth += 1
            if topic_correct:
                correct_topic += 1
                
            total_time += inference_time
            
            result = {
                'id': sample['id'],
                'statement': sample['statement'],
                'expected_truth': sample['expected_truth'],
                'expected_topic': sample['expected_topic'],
                'predicted_truth': predicted_truth,
                'predicted_topic': predicted_topic,
                'truth_correct': truth_correct,
                'topic_correct': topic_correct,
                'inference_time': inference_time
            }
            
            results.append(result)
            
            # Print result
            truth_status = "✅" if truth_correct else "❌"
            topic_status = "✅" if topic_correct else "❌"
            print(f"  {truth_status} Truth: {predicted_truth} (expected {sample['expected_truth']})")
            print(f"  {topic_status} Topic: {predicted_topic} (expected {sample['expected_topic']})")
            print(f"  ⏱️  Time: {inference_time:.2f}s")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
            continue
    
    # Calculate metrics
    total_samples = len(results)
    if total_samples == 0:
        return {
            'total_samples': 0,
            'truth_accuracy': 0,
            'topic_accuracy': 0,
            'overall_accuracy': 0,
            'avg_time': 0,
            'total_time': 0
        }
    
    truth_accuracy = correct_truth / total_samples
    topic_accuracy = correct_topic / total_samples
    overall_accuracy = (correct_truth + correct_topic) / (total_samples * 2)
    avg_time = total_time / total_samples
    
    return {
        'total_samples': total_samples,
        'truth_accuracy': truth_accuracy,
        'topic_accuracy': topic_accuracy,
        'overall_accuracy': overall_accuracy,
        'avg_time': avg_time,
        'total_time': total_time,
        'results': results
    }

def print_summary(metrics: Dict):
    """
    Print a comprehensive summary of the evaluation results
    """
    print("=" * 80)
    print("📊 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Truth Accuracy: {metrics['truth_accuracy']:.1%} ({metrics['truth_accuracy']*metrics['total_samples']:.0f}/{metrics['total_samples']})")
    print(f"Topic Accuracy: {metrics['topic_accuracy']:.1%} ({metrics['topic_accuracy']*metrics['total_samples']:.0f}/{metrics['total_samples']})")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.1%} ({(metrics['truth_accuracy'] + metrics['topic_accuracy'])*metrics['total_samples']:.0f}/{metrics['total_samples']*2})")
    print(f"Average Time per Sample: {metrics['avg_time']:.2f}s")
    print(f"Total Time: {metrics['total_time']:.2f}s")
    print(f"Score / 20: {(metrics['truth_accuracy'] + metrics['topic_accuracy'])*metrics['total_samples']:.1f}/20")
    print("=" * 80)

def main():
    """
    Main evaluation function
    """
    # Load first 10 samples from train data
    samples = load_train_data(n_samples=10)
    
    if not samples:
        print("❌ No samples loaded. Check if train data exists.")
        return
    
    print(f"📚 Loaded {len(samples)} samples from train data")
    print()
    
    # Evaluate the model
    metrics = evaluate_model(samples)
    
    # Print summary
    print_summary(metrics)
    
    # Save detailed results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"📄 Detailed results saved to {output_file}")

if __name__ == "__main__":
    main() 