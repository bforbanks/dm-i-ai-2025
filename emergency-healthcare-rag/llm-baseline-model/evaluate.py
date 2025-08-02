import json
import time
from pathlib import Path
import sys
import os

# Add parent directory to path to import utils and model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import load_statement_sample

def evaluate_llm_baseline(max_examples: int = 20):
    """Simple evaluation of LLM-only baseline"""
    
    print("\n🤖 LLM-ONLY BASELINE EVALUATION")
    print("=" * 60)
    print()
    
    # Load topics for ID mapping
    topics_file = Path("data/topics.json")
    with open(topics_file, 'r') as f:
        topics_data = json.load(f)
        topic_to_id = topics_data
        id_to_topic = {v: k for k, v in topics_data.items()}
    
    # Get all statement files
    statements_dir = Path("data/train/statements")
    statement_files = sorted([f for f in statements_dir.glob("*.txt") if f.name.startswith("statement_")])
    
    if max_examples:
        statement_files = statement_files[:max_examples]
    
    total_examples = len(statement_files)
    correct_truth = 0
    correct_topic = 0
    
    # Timing
    total_time = 0
    
    for i, statement_file in enumerate(statement_files):
        statement_id = statement_file.stem.split('_')[1]
        
        try:
            # Load statement and ground truth
            statement, true_answer = load_statement_sample(statement_id)
            true_truth = true_answer['statement_is_true']
            true_topic = true_answer['statement_topic']
            true_topic_name = id_to_topic.get(true_topic, f"Unknown (ID: {true_topic})")
            
            print(f"📝 STATEMENT {statement_id}")
            print(f"Text: {statement}")
            print()
            
            # Time prediction
            start_time = time.time()
            pred_truth, pred_topic_name = predict(statement)
            pred_time = time.time() - start_time
            total_time += pred_time
            
            # Convert topic name to ID for comparison
            pred_topic_id = topic_to_id.get(pred_topic_name, -1)
            
            # Check correctness
            truth_correct = pred_truth == true_truth
            topic_correct = pred_topic_id == true_topic
            
            print(f"⏱️  Timing: {pred_time:.3f}s")
            print()
            print(f"🎯 PREDICTIONS:")
            print(f"   Truth: {pred_truth} {'✅ CORRECT' if truth_correct else '❌ WRONG'} (should be: {true_truth})")
            print(f"   Topic: {pred_topic_name} {'✅ CORRECT' if topic_correct else '❌ WRONG'}")
            print(f"   Should be: {true_topic_name}")
            print()
            
            # Update counters
            if truth_correct:
                correct_truth += 1
            if topic_correct:
                correct_topic += 1
                
        except Exception as e:
            print(f"ERROR processing {statement_id}: {e}")
        
        print("-" * 60)
        print()
    
    # Final results
    print(f"📊 FINAL RESULTS ({total_examples} examples)")
    print("=" * 60)
    print(f"✅ Truth Accuracy:   {correct_truth}/{total_examples} ({100*correct_truth/total_examples:.1f}%)")
    print(f"🎯 Topic Accuracy:   {correct_topic}/{total_examples} ({100*correct_topic/total_examples:.1f}%)")
    print()
    
    # Overall score
    total_points = correct_truth + correct_topic
    max_points = total_examples * 2
    overall_percentage = 100 * total_points / max_points
    print(f"🏆 OVERALL SCORE:")
    print(f"   Combined:           {total_points}/{max_points} ({overall_percentage:.1f}%)")
    print()
    
    # Timing
    avg_time = total_time / total_examples if total_examples > 0 else 0
    print(f"⏱️  AVERAGE TIMING:")
    print(f"   LLM Prediction: {avg_time:.3f}s")
    print("=" * 60)

if __name__ == "__main__":
    # Import the model
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import predict
    
    max_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    evaluate_llm_baseline(max_examples) 