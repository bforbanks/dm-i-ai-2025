import warnings
warnings.filterwarnings("ignore")
# Suppress specific torch warning
import logging
logging.getLogger("transformers.models.bert.modeling_bert").setLevel(logging.ERROR)
logging.getLogger("torch.nn.modules.module").setLevel(logging.ERROR)

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path to import utils and model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from utils import load_statement_sample
import importlib
import importlib.util

# Import the main model.py (not model-1/model.py)
model_spec = importlib.util.spec_from_file_location("main_model", os.path.join(parent_dir, "model.py"))
main_model = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(main_model)
predict = main_model.predict

# Load topic mappings
topics_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'topics.json')
with open(topics_file, 'r') as f:
    topics_data = json.load(f)
    id_to_topic = {v: k for k, v in topics_data.items()}



def get_detailed_semantic_analysis_with_timing(statement: str, true_topic_id: int = None):
    """Get detailed semantic search analysis with TRUE embedding vs search separation"""
    # Import combined-model-1 search functions
    search_module = importlib.import_module("combined-model-1.search")
    
    # Load data and model (no timing)
    data = search_module.load_embeddings()
    from sentence_transformers import SentenceTransformer
    if 'model_path' in data:
        model = SentenceTransformer(data['model_path'])
    else:
        model = search_module.load_local_model()
    
    # Time JUST the statement embedding
    embedding_start = time.time()
    statement_embedding = model.encode([statement])
    embedding_time = time.time() - embedding_start
    
    # Time JUST the search operations (using pre-computed embedding)
    search_start = time.time()
    all_topics = search_module.get_top_k_topics_with_context_from_embedding(statement_embedding, k=115)
    search_time = time.time() - search_start
    
    # Get context and file sources (no timing needed)
    chunk_files = set()
    context = ""
    if all_topics:
        chosen_topic_id = all_topics[0]['topic_id']
        context = search_module.get_targeted_context_for_topic(statement, chosen_topic_id, max_chars=1500)
        
        # Extract file sources
        for topic in all_topics[:1]:  # Just for the chosen topic
            for chunk in topic['best_chunks']:
                chunk_text = chunk['text']
                # Find which files these chunks came from
                for i, stored_chunk in enumerate(data['chunks']):
                    if chunk_text[:100] in stored_chunk or stored_chunk[:100] in chunk_text:
                        file_path = data['metadata'][i]
                        chunk_files.add(Path(file_path).name)
                        break
    
    context_preview = context[:300] + "..." if len(context) > 300 else context
    return all_topics, list(chunk_files), context_preview, embedding_time, search_time

def get_topics_to_display(all_topics, true_topic_id, pred_topic_id):
    """Get the topics to display, ensuring true topic is included even if not in top 5"""
    top_5 = all_topics[:5]
    topics_to_show = []
    
    # Add top 5 with their rankings
    for i, topic in enumerate(top_5):
        topics_to_show.append({
            **topic, 
            'display_rank': i + 1,
            'is_top_5': True
        })
    
    # Check if true topic is in top 5
    true_topic_in_top_5 = any(topic['topic_id'] == true_topic_id for topic in top_5)
    
    # If true topic is not in top 5, find it and add it
    if not true_topic_in_top_5 and true_topic_id is not None:
        for i, topic in enumerate(all_topics):
            if topic['topic_id'] == true_topic_id:
                topics_to_show.append({
                    **topic,
                    'display_rank': i + 1,
                    'is_top_5': False
                })
                break
    
    return topics_to_show

def evaluate_detailed(max_examples: int = 20):
    """Detailed evaluation showing all decision components"""
    
    print("\n🔬 DETAILED MODEL EVALUATION")
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
    
    # Timing accumulation (single measurement per operation)
    embedding_times = []
    search_times = []
    model_times = []
    individual_timings = []
    
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
            
            # Get detailed semantic analysis with embedding timing breakdown
            print(f"🔍 PERFORMANCE TIMING:")
            
            # Get timing breakdown including separate embedding measurement
            all_topics, source_files, context_preview, embedding_time, search_time = get_detailed_semantic_analysis_with_timing(statement, true_topic)
            
            # Time model prediction separately
            model_start = time.time()
            pred_truth, pred_topic = predict(statement)
            model_time = time.time() - model_start
            
            total_time = embedding_time + search_time + model_time
            
            print(f"   Statement Embedding: {embedding_time:.3f}s")
            print(f"   Semantic Search: {search_time:.3f}s") 
            print(f"   Model Prediction: {model_time:.3f}s")
            print(f"   Total: {total_time:.3f}s")
            print()
            
            # Store timing data for statistics
            timing_data = {
                'statement_id': statement_id,
                'embedding_time': embedding_time,
                'search_time': search_time,
                'model_time': model_time,
                'total_time': total_time
            }
            individual_timings.append(timing_data)
            
            # Accumulate for statistics
            embedding_times.append(embedding_time)
            search_times.append(search_time)
            model_times.append(model_time)
            
            # Show results with VERY CLEAR correctness indicators
            truth_correct = pred_truth == true_truth
            topic_correct = pred_topic == true_topic
            
            print(f"🎯 PREDICTIONS:")
            print(f"   Truth: {pred_truth} {'✅ CORRECT' if truth_correct else '❌ WRONG'} (should be: {true_truth})")
            print(f"   Topic: {pred_topic} ({id_to_topic.get(pred_topic, 'Unknown')}) {'✅ CORRECT' if topic_correct else '❌ WRONG'}")
            print(f"   Should be: {id_to_topic.get(true_topic, 'Unknown')} (ID: {true_topic})")
            print()
            
            # CLEAR STATUS - SEPARATE COMPONENT SCORING
            print("📊 COMPONENT RESULTS (scored independently):")
            truth_status = "✅ CORRECT" if truth_correct else "❌ WRONG"
            topic_status = "✅ CORRECT" if topic_correct else "❌ WRONG"
            print(f"   Truth Component: {truth_status}")
            print(f"   Topic Component: {topic_status}")
            
            points_earned = (1 if truth_correct else 0) + (1 if topic_correct else 0)
            print(f"   Points: {points_earned}/2")
            print()
            
            # Get topics to display including true topic even if not in top 5
            topics_to_display = get_topics_to_display(all_topics, true_topic, pred_topic)
            
            # Find actual ranking of true topic
            true_topic_rank = None
            for i, topic in enumerate(all_topics):
                if topic['topic_id'] == true_topic:
                    true_topic_rank = i + 1
                    break
                    
            print(f"🔍 SEMANTIC SEARCH TOPICS (True topic rank: {true_topic_rank}/115):")
            
            # Display top 5 first
            top_5_topics = [t for t in topics_to_display if t['is_top_5']]
            for topic in top_5_topics:
                marker = "🎯" if topic['topic_id'] == true_topic else "📍" if topic['topic_id'] == pred_topic else "  "
                score = topic.get('score', 0.0)
                print(f"   {topic['display_rank']}. {marker} {topic['topic_name']} (ID: {topic['topic_id']}, Score: {score:.3f})")
            
            # Display true topic if it's not in top 5
            non_top_5_topics = [t for t in topics_to_display if not t['is_top_5']]
            if non_top_5_topics:
                print(f"   ...")
                for topic in non_top_5_topics:
                    marker = "🎯" if topic['topic_id'] == true_topic else "📍" if topic['topic_id'] == pred_topic else "  "
                    score = topic.get('score', 0.0)
                    print(f"   {topic['display_rank']}. {marker} {topic['topic_name']} (ID: {topic['topic_id']}, Score: {score:.3f})")
            
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
            
            # Timing section removed since we now show it earlier
            
            # Update accuracy counters
            if truth_correct:
                correct_truth += 1
            if topic_correct:
                correct_topic += 1
                
        except Exception as e:
            print(f"ERROR processing {statement_id}: {e}")
        
        print()
        print("-" * 80)
        print()
    
    # Final statistics with emphasis on separate scoring
    print(f"📊 FINAL RESULTS ({total_examples} examples)")
    print("=" * 80)
    print(f"📈 INDIVIDUAL COMPONENT SCORES (Each scored separately):")
    print(f"   ✅ Truth Accuracy:   {correct_truth}/{total_examples} ({100*correct_truth/total_examples:.1f}%)")
    print(f"   🎯 Topic Accuracy:   {correct_topic}/{total_examples} ({100*correct_topic/total_examples:.1f}%)")
    print()
    
    # Overall score: (truth + topic) / (2 * examples)
    total_points = correct_truth + correct_topic
    max_points = total_examples * 2
    overall_percentage = 100 * total_points / max_points
    print(f"🏆 OVERALL SCORE:")
    print(f"   Combined:           {total_points}/{max_points} ({overall_percentage:.1f}%)")
    print()
    
    # Average timing statistics
    if total_examples > 0 and search_times:
        avg_embedding = sum(embedding_times) / len(embedding_times)
        avg_search = sum(search_times) / len(search_times)
        avg_model = sum(model_times) / len(model_times)
        avg_total = avg_embedding + avg_search + avg_model
        
        print(f"⏱️  AVERAGE TIMING:")
        print(f"   Statement Embedding: {avg_embedding:.3f}s")
        print(f"   Semantic Search: {avg_search:.3f}s")
        print(f"   Model Prediction: {avg_model:.3f}s")
        print(f"   Total Average: {avg_total:.3f}s")
    
    print("=" * 80)

if __name__ == "__main__":
    import sys
    max_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    evaluate_detailed(max_examples)