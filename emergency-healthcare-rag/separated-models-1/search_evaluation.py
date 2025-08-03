#!/usr/bin/env python3
"""
Search evaluation script for separated-models-1
Tests topic selection accuracy and context quality without running LLM
"""

import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import statistics

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from search import hybrid_search, get_best_topic, get_rich_context_for_statement
from config import get_model_info

def load_all_train_data() -> List[Dict]:
    """Load all 200 samples from train data"""
    statements_dir = Path("data/train/statements")
    answers_dir = Path("data/train/answers")
    
    samples = []
    for i in range(200):  # All 200 samples
        statement_file = statements_dir / f"statement_{i:04d}.txt"
        answer_file = answers_dir / f"statement_{i:04d}.json"
        
        if statement_file.exists() and answer_file.exists():
            with open(statement_file, 'r') as f:
                statement = f.read().strip()
            
            with open(answer_file, 'r') as f:
                answer_data = json.load(f)
            
            samples.append({
                'statement': statement,
                'expected_truth': answer_data['statement_is_true'],
                'expected_topic': answer_data['statement_topic'],
                'sample_id': i
            })
    
    return samples

def load_topics_mapping() -> Dict[str, int]:
    """Load topic name to ID mapping"""
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def get_topic_name(topic_id: int) -> str:
    """Get topic name from ID"""
    topics_data = load_topics_mapping()
    for name, tid in topics_data.items():
        if tid == topic_id:
            return name
    return f"Unknown_{topic_id}"

def evaluate_search_accuracy(samples: List[Dict]) -> Dict:
    """Evaluate search accuracy and context quality"""
    results = []
    topic_accuracy = 0
    topic_in_top_3 = 0
    topic_in_top_5 = 0
    topic_in_top_10 = 0
    
    print("🔍 Evaluating search accuracy and context quality...")
    print("=" * 80)
    
    for i, sample in enumerate(samples, 1):
        statement = sample['statement']
        expected_topic = sample['expected_topic']
        expected_topic_name = get_topic_name(expected_topic)
        
        print(f"\nSample {i}/{len(samples)}: {statement[:60]}...")
        
        # Time the search
        start_time = time.time()
        search_results = hybrid_search(statement, top_k=10)
        search_time = time.time() - start_time
        
        # Get best topic
        predicted_topic = get_best_topic(statement)
        predicted_topic_name = get_topic_name(predicted_topic)
        
        # Check if expected topic is in top results
        found_topics = [result['topic_id'] for result in search_results]
        topic_rank = -1
        if expected_topic in found_topics:
            topic_rank = found_topics.index(expected_topic) + 1
        
        # Update counters
        if predicted_topic == expected_topic:
            topic_accuracy += 1
        
        if expected_topic in found_topics[:3]:
            topic_in_top_3 += 1
        if expected_topic in found_topics[:5]:
            topic_in_top_5 += 1
        if expected_topic in found_topics[:10]:
            topic_in_top_10 += 1
        
        # Get context for analysis
        context = get_rich_context_for_statement(statement, predicted_topic)
        context_length = len(context)
        
        # Print results
        topic_correct = "✅" if predicted_topic == expected_topic else "❌"
        print(f"  {topic_correct} Predicted: {predicted_topic_name} (ID: {predicted_topic})")
        print(f"     Expected: {expected_topic_name} (ID: {expected_topic})")
        print(f"     Topic rank in search: {topic_rank if topic_rank > 0 else 'Not found'}")
        print(f"     Search time: {search_time:.3f}s")
        print(f"     Context length: {context_length} chars")
        
        # Show top 3 search results
        print("     Top 3 search results:")
        for j, result in enumerate(search_results[:3], 1):
            result_topic = get_topic_name(result['topic_id'])
            print(f"       {j}. {result_topic} (score: {result['score']:.4f})")
        
        # Show context preview
        context_preview = context[:200] + "..." if len(context) > 200 else context
        print(f"     Context preview: {context_preview}")
        
        results.append({
            'sample_id': sample['sample_id'],
            'statement': statement,
            'expected_topic': expected_topic,
            'expected_topic_name': expected_topic_name,
            'predicted_topic': predicted_topic,
            'predicted_topic_name': predicted_topic_name,
            'topic_correct': predicted_topic == expected_topic,
            'topic_rank': topic_rank,
            'search_time': search_time,
            'context_length': context_length,
            'top_results': search_results[:5]  # Store top 5 for analysis
        })
    
    # Calculate statistics
    total_samples = len(samples)
    accuracy = topic_accuracy / total_samples
    top_3_rate = topic_in_top_3 / total_samples
    top_5_rate = topic_in_top_5 / total_samples
    top_10_rate = topic_in_top_10 / total_samples
    
    avg_search_time = statistics.mean([r['search_time'] for r in results])
    avg_context_length = statistics.mean([r['context_length'] for r in results])
    
    print("\n" + "=" * 80)
    print("📊 SEARCH EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total samples: {total_samples}")
    print(f"Topic accuracy: {accuracy:.1%} ({topic_accuracy}/{total_samples})")
    print(f"Topic in top 3: {top_3_rate:.1%} ({topic_in_top_3}/{total_samples})")
    print(f"Topic in top 5: {top_5_rate:.1%} ({topic_in_top_5}/{total_samples})")
    print(f"Topic in top 10: {top_10_rate:.1%} ({topic_in_top_10}/{total_samples})")
    print(f"Average search time: {avg_search_time:.3f}s")
    print(f"Average context length: {avg_context_length:.0f} chars")
    
    # Analyze failures
    failures = [r for r in results if not r['topic_correct']]
    if failures:
        print(f"\n❌ TOPIC SELECTION FAILURES ({len(failures)} cases):")
        for failure in failures[:10]:  # Show first 10 failures
            print(f"  Sample {failure['sample_id']}: '{failure['statement'][:50]}...'")
            print(f"    Expected: {failure['expected_topic_name']}")
            print(f"    Predicted: {failure['predicted_topic_name']}")
            print(f"    Rank in search: {failure['topic_rank']}")
            print()
    
    return {
        'total_samples': total_samples,
        'topic_accuracy': accuracy,
        'topic_in_top_3': top_3_rate,
        'topic_in_top_5': top_5_rate,
        'topic_in_top_10': top_10_rate,
        'avg_search_time': avg_search_time,
        'avg_context_length': avg_context_length,
        'results': results
    }

def save_results(results: Dict, filename: str = "search_evaluation_results.json"):
    """Save evaluation results to file"""
    # Remove large data from results for storage
    storage_results = results.copy()
    for result in storage_results['results']:
        result.pop('top_results', None)  # Remove large search results
    
    with open(filename, 'w') as f:
        json.dump(storage_results, f, indent=2)
    print(f"\n💾 Results saved to {filename}")

def main():
    """Main evaluation function"""
    print("🔍 Search Evaluation for Separated Models 1")
    print("Testing topic selection accuracy and context quality")
    print("=" * 80)
    
    # Load all train data
    print("📚 Loading train data...")
    samples = load_all_train_data()
    
    if not samples:
        print("❌ No samples loaded. Check if train data exists.")
        return
    
    print(f"📚 Loaded {len(samples)} samples from train data")
    
    # Run evaluation
    results = evaluate_search_accuracy(samples)
    
    # Save results
    save_results(results)
    
    print("\n✅ Search evaluation complete!")

if __name__ == "__main__":
    main() 