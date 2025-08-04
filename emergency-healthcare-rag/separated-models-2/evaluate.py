#!/usr/bin/env python3
"""
Evaluation script for separated-models-2
Tests BM25-only search performance
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import re
from search import bm25_search, get_best_topic
from model import create_rag_model

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate separated-models-2 RAG system')
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to evaluate')
    parser.add_argument('--model', type=str, default=None, help='LLM model to use')
    parser.add_argument('--search-only', action='store_true', help='Evaluate only search component')
    parser.add_argument('--full-pipeline', action='store_true', help='Evaluate only full pipeline')
    return parser.parse_args()

def load_test_data() -> List[Tuple[str, int, bool]]:
    """Load test statements with their true topic and truth value"""
    statements = []
    statement_dir = Path("data/train/statements")
    answer_dir = Path("data/train/answers")
    
    for path in sorted(statement_dir.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((answer_dir / f"statement_{sid}.json").read_text())
        
        statements.append((
            statement,
            ans["statement_topic"],
            bool(ans["statement_is_true"])
        ))
    
    return statements

def extract_truth_from_response(response: str) -> bool:
    """Extract TRUE/FALSE from LLM response (expects only a number: 1 for TRUE, 0 for FALSE)"""
    response_text = response.strip()
    
    # Handle direct numeric responses
    if response_text in ['0', '1']:
        return int(response_text) == 1
    
    # Fallback: try to extract number from response
    import re
    numbers = re.findall(r'\d+', response_text)
    if numbers:
        return int(numbers[0]) == 1
        
    # Default fallback
    print(f"Warning: Could not parse response: {response_text}")
    return False  # Default to false for safety

def evaluate_search_only():
    """Evaluate only the search component (BM25) - optimized version"""
    print("🔍 EVALUATING BM25 SEARCH COMPONENT")
    print("=" * 60)
    
    statements = load_test_data()
    print(f"Testing on {len(statements)} statements")
    
    # Do one search per statement and check all top-k values
    search_results = {}
    top_k_values = range(1, 11)
    
    # Initialize counters for each top-k
    for top_k in top_k_values:
        search_results[top_k] = 0
    
    for stmt, true_topic, _ in tqdm(statements, desc="Search evaluation"):
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

def evaluate_full_pipeline(model_name: str = None):
    """Evaluate the full RAG pipeline"""
    print("🤖 EVALUATING FULL RAG PIPELINE")
    print("=" * 60)
    
    statements = load_test_data()
    print(f"Testing on {len(statements)} statements")
    
    # Create RAG model with specified model
    rag_model = create_rag_model(model_name)
    print(f"Using model: {rag_model.get_model_info()['llm_model']}")
    
    # Evaluation metrics
    truth_correct = 0
    topic_correct = 0
    total_time = 0
    
    results = []
    
    for i, (statement, true_topic, true_truth) in enumerate(tqdm(statements, desc="Processing")):
        start_time = time.time()
        
        # Process through RAG pipeline
        result = rag_model.process_statement(statement)
        
        end_time = time.time()
        processing_time = end_time - start_time
        total_time += processing_time
        
        # Extract predicted truth
        predicted_truth = extract_truth_from_response(result['response'])
        
        # Check accuracy
        topic_accuracy = (result['topic_id'] == true_topic)
        truth_accuracy = (predicted_truth == true_truth)
        
        if topic_accuracy:
            topic_correct += 1
        if truth_accuracy:
            truth_correct += 1
        
        # Store detailed results
        results.append({
            'statement': statement,
            'true_topic': true_topic,
            'predicted_topic': result['topic_id'],
            'true_truth': true_truth,
            'predicted_truth': predicted_truth,
            'response': result['response'],
            'topic_correct': topic_accuracy,
            'truth_correct': truth_accuracy,
            'processing_time': processing_time
        })
    
    # Calculate metrics
    total_samples = len(statements)
    truth_accuracy = truth_correct / total_samples
    topic_accuracy = topic_correct / total_samples
    overall_accuracy = (truth_correct + topic_correct) / (total_samples * 2)
    avg_time = total_time / total_samples
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Samples: {total_samples}")
    print(f"Truth Accuracy: {truth_accuracy:.1%} ({truth_correct}/{total_samples})")
    print(f"Topic Accuracy: {topic_accuracy:.1%} ({topic_correct}/{total_samples})")
    print(f"Overall Accuracy: {overall_accuracy:.1%} ({truth_correct + topic_correct}/{total_samples * 2})")
    print(f"Average Time per Sample: {avg_time:.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Score / 400: {int((truth_correct + topic_correct) / 2)}/400")
    print("=" * 60)
    
    # Save detailed results
    output_file = "separated-models-2/search_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            'summary': {
                'total_samples': total_samples,
                'truth_accuracy': truth_accuracy,
                'topic_accuracy': topic_accuracy,
                'overall_accuracy': overall_accuracy,
                'avg_time': avg_time,
                'total_time': total_time,
                'score_400': int((truth_correct + topic_correct) / 2)
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    
    return {
        'truth_accuracy': truth_accuracy,
        'topic_accuracy': topic_accuracy,
        'overall_accuracy': overall_accuracy,
        'avg_time': avg_time,
        'results': results
    }

def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    print("🚑 EMERGENCY HEALTHCARE RAG - SEPARATED MODELS 2 EVALUATION")
    print("=" * 80)
    print("BM25-only search with optimized chunking (chunk_size=128, overlap=12)")
    if args.model:
        print(f"Using model: {args.model}")
    print("=" * 80)
    
    # Determine what to evaluate
    if args.search_only:
        # Evaluate search component only
        search_results = evaluate_search_only()
    elif args.full_pipeline:
        # Evaluate full pipeline only
        pipeline_results = evaluate_full_pipeline(args.model)
    else:
        # Evaluate both (default)
        search_results = evaluate_search_only()
        print("\n")
        pipeline_results = evaluate_full_pipeline(args.model)
    
    print("\n" + "=" * 80)
    print("✅ EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main() 