import warnings
warnings.filterwarnings("ignore")
# Suppress specific torch warning
import logging
logging.getLogger("transformers.models.bert.modeling_bert").setLevel(logging.ERROR)
logging.getLogger("torch.nn.modules.module").setLevel(logging.ERROR)

import os
import json
import time
from pathlib import Path
from utils import load_statement_sample
from model import predict
import importlib

# Load topic mappings
with open('data/topics.json', 'r') as f:
    topics_data = json.load(f)
    id_to_topic = {v: k for k, v in topics_data.items()}

def get_detailed_semantic_analysis(statement: str):
    """Get detailed semantic search analysis including file sources"""
    # Import model-1 search functions
    search_module = importlib.import_module("model-1.search")
    
    # Get top 5 topic candidates with context
    top_topics = search_module.get_top_k_topics_with_context(statement, k=5)
    
    # Also get the targeted context to see what chunks are fed to LLM
    if top_topics:
        chosen_topic_id = top_topics[0]['topic_id']
        context = search_module.get_targeted_context_for_topic(statement, chosen_topic_id, max_chars=1500)
        
        # Extract file sources from chunks
        data = search_module.load_embeddings()
        chunk_files = set()
        for topic in top_topics[:1]:  # Just for the chosen topic
            for chunk in topic['best_chunks']:
                chunk_text = chunk['text']
                # Find which files these chunks came from
                for i, stored_chunk in enumerate(data['chunks']):
                    if chunk_text[:100] in stored_chunk or stored_chunk[:100] in chunk_text:
                        file_path = data['metadata'][i]
                        chunk_files.add(Path(file_path).name)
                        break
        
        return top_topics, list(chunk_files), context[:300] + "..." if len(context) > 300 else context
    
    return [], [], ""

def evaluate_detailed(max_examples: int = 5):
    """Detailed evaluation showing all decision components"""
    
    print("🔬 DETAILED MODEL EVALUATION")
    print("=" * 80)
    print()
    
    # Get all statement files
    statements_dir = Path("data/train/statements")
    statement_files = sorted([f for f in statements_dir.glob("*.txt") if f.name.startswith("statement_")])
    
    if max_examples:
        statement_files = statement_files[:max_examples]
    
    total_examples = len(statement_files)
    correct_truth = 0
    correct_topic = 0
    
    for i, statement_file in enumerate(statement_files):
        # Extract ID from filename (statement_0000.txt -> 0000)
        statement_id = statement_file.stem.split('_')[1]
        
        try:
            # Load statement and ground truth
            statement, true_answer = load_statement_sample(statement_id)
            true_truth = true_answer['statement_is_true']
            true_topic = true_answer['statement_topic']
            
            print(f"📝 STATEMENT {statement_id}")
            print(f"Text: {statement}")
            print()
            
            # Get detailed semantic analysis
            start_time = time.time()
            top_5_topics, source_files, context_preview = get_detailed_semantic_analysis(statement)
            analysis_time = time.time() - start_time
            
            # Get model prediction
            pred_start = time.time()
            pred_truth, pred_topic = predict(statement)
            pred_time = time.time() - pred_start
            
            # Show results with VERY CLEAR correctness indicators
            truth_correct = pred_truth == true_truth
            topic_correct = pred_topic == true_topic
            both_correct = truth_correct and topic_correct
            
            print(f"🎯 PREDICTIONS:")
            print(f"   Truth: {pred_truth} {'✅ CORRECT' if truth_correct else '❌ WRONG'} (should be: {true_truth})")
            print(f"   Topic: {pred_topic} ({id_to_topic.get(pred_topic, 'Unknown')}) {'✅ CORRECT' if topic_correct else '❌ WRONG'}")
            print(f"   Should be: {id_to_topic.get(true_topic, 'Unknown')} (ID: {true_topic})")
            print()
            
            # CLEAR OVERALL STATUS
            if both_correct:
                print("🏆 RESULT: ✅ BOTH TRUTH AND TOPIC CORRECT! 🏆")
            elif truth_correct:
                print("🔶 RESULT: ✅ TRUTH CORRECT, ❌ TOPIC WRONG")
            elif topic_correct:
                print("🔶 RESULT: ❌ TRUTH WRONG, ✅ TOPIC CORRECT")
            else:
                print("🔴 RESULT: ❌ BOTH TRUTH AND TOPIC WRONG")
            print()
            
            print(f"🔍 TOP 5 SEMANTIC SEARCH TOPICS:")
            for j, topic in enumerate(top_5_topics[:5]):
                marker = "🎯" if topic['topic_id'] == true_topic else "📍" if topic['topic_id'] == pred_topic else "  "
                score = topic.get('combined_score', 0.0)
                print(f"   {j+1}. {marker} {topic['topic_name']} (ID: {topic['topic_id']}, Score: {score:.3f})")
            print()
            
            print(f"📄 CONTEXT SOURCES (from {len(source_files)} files):")
            for file in sorted(source_files):
                print(f"   • {file}")
            if not source_files:
                print("   • No specific files identified")
            print()
            
            print(f"📖 CONTEXT PREVIEW (first 300 chars):")
            print(f"   {context_preview}")
            print()
            
            print(f"⏱️  TIMING:")
            print(f"   Semantic Analysis: {analysis_time:.1f}s")
            print(f"   Model Prediction: {pred_time:.1f}s")
            print(f"   Total: {analysis_time + pred_time:.1f}s")
            
            # Update accuracy counters
            if pred_truth == true_truth:
                correct_truth += 1
            if pred_topic == true_topic:
                correct_topic += 1
                
        except Exception as e:
            print(f"❌ ERROR processing {statement_id}: {e}")
        
        print()
        print("-" * 80)
        print()
    
    # Final statistics with CLEAR emphasis
    both_correct_count = min(correct_truth, correct_topic)
    
    print(f"📊 FINAL RESULTS ({total_examples} examples)")
    print("=" * 60)
    print(f"🏆 BOTH CORRECT:     {both_correct_count}/{total_examples} ({100*both_correct_count/total_examples:.1f}%)")
    print(f"✅ Truth Accuracy:   {correct_truth}/{total_examples} ({100*correct_truth/total_examples:.1f}%)")
    print(f"🎯 Topic Accuracy:   {correct_topic}/{total_examples} ({100*correct_topic/total_examples:.1f}%)")
    
    if both_correct_count == total_examples:
        print("🎉 PERFECT SCORE! ALL PREDICTIONS CORRECT! 🎉")
    elif both_correct_count >= total_examples * 0.8:
        print("🌟 EXCELLENT PERFORMANCE! >80% fully correct")
    elif both_correct_count >= total_examples * 0.6:
        print("👍 GOOD PERFORMANCE! >60% fully correct")
    else:
        print("⚠️  NEEDS IMPROVEMENT - <60% fully correct")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    max_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    evaluate_detailed(max_examples)