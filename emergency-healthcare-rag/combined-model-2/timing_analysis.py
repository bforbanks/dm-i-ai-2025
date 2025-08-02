#!/usr/bin/env python3
"""
Timing analysis script to identify bottlenecks in the pipeline
"""

import sys
import time
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from search import get_top_k_topics_with_context, get_targeted_context_for_topic
from llm import classify_truth_and_topic_combined

def analyze_timing():
    """
    Analyze timing of each component in the pipeline
    """
    test_statement = "Coronary heart disease affects approximately 15.5 million people in the United States."
    
    print("🔍 TIMING ANALYSIS")
    print("=" * 60)
    
    # Step 1: Search timing
    print("1. SEARCH COMPONENT")
    start_time = time.time()
    candidates = get_top_k_topics_with_context(test_statement, k=8)
    search_time = time.time() - start_time
    print(f"   ⏱️  Search time: {search_time:.2f}s")
    print(f"   📊 Found {len(candidates)} candidates")
    
    # Step 2: Context retrieval timing
    print("\n2. CONTEXT RETRIEVAL")
    start_time = time.time()
    context = get_targeted_context_for_topic(test_statement, candidates[0]['topic_id'], max_chars=2500)
    context_time = time.time() - start_time
    print(f"   ⏱️  Context retrieval time: {context_time:.2f}s")
    print(f"   📄 Context length: {len(context)} characters")
    
    # Step 3: LLM timing
    print("\n3. LLM CLASSIFICATION")
    start_time = time.time()
    try:
        truth, topic = classify_truth_and_topic_combined(test_statement, candidates, context)
        llm_time = time.time() - start_time
        print(f"   ⏱️  LLM time: {llm_time:.2f}s")
        print(f"   🎯 Predicted: Truth={truth}, Topic={topic}")
    except Exception as e:
        print(f"   ❌ LLM error: {e}")
        llm_time = 0
    
    # Step 4: Total timing
    total_time = search_time + context_time + llm_time
    print(f"\n📊 TOTAL TIMING BREAKDOWN")
    print(f"   Search: {search_time:.2f}s ({search_time/total_time*100:.1f}%)")
    print(f"   Context: {context_time:.2f}s ({context_time/total_time*100:.1f}%)")
    print(f"   LLM: {llm_time:.2f}s ({llm_time/total_time*100:.1f}%)")
    print(f"   Total: {total_time:.2f}s")
    
    # Step 5: Bottleneck identification
    print(f"\n🎯 BOTTLENECK ANALYSIS")
    if llm_time > search_time + context_time:
        print("   🐌 LLM is the bottleneck (>50% of total time)")
        print("   💡 Solutions: Use larger GPU, smaller model, or parallel processing")
    elif search_time > context_time + llm_time:
        print("   🐌 Search is the bottleneck (>50% of total time)")
        print("   💡 Solutions: Cache embeddings, optimize search algorithm")
    elif context_time > search_time + llm_time:
        print("   🐌 Context retrieval is the bottleneck (>50% of total time)")
        print("   💡 Solutions: Cache context, reduce context size")
    else:
        print("   ⚖️  Time is relatively balanced across components")
    
    return {
        'search_time': search_time,
        'context_time': context_time,
        'llm_time': llm_time,
        'total_time': total_time
    }

def test_multiple_statements():
    """
    Test timing across multiple statements to get average
    """
    test_statements = [
        "Coronary heart disease affects approximately 15.5 million people in the United States.",
        "The ureter receives segmental blood supply from the renal arteries proximally.",
        "Intravesicular catheter pressure measurement using a Foley catheter provides accurate assessment."
    ]
    
    print("\n📈 MULTIPLE STATEMENT TIMING")
    print("=" * 60)
    
    total_search = 0
    total_context = 0
    total_llm = 0
    total_overall = 0
    
    for i, statement in enumerate(test_statements, 1):
        print(f"\n--- Statement {i} ---")
        
        # Search
        start = time.time()
        candidates = get_top_k_topics_with_context(statement, k=8)
        search_time = time.time() - start
        
        # Context
        start = time.time()
        context = get_targeted_context_for_topic(statement, candidates[0]['topic_id'], max_chars=2500)
        context_time = time.time() - start
        
        # LLM
        start = time.time()
        try:
            truth, topic = classify_truth_and_topic_combined(statement, candidates, context)
            llm_time = time.time() - start
        except Exception as e:
            llm_time = 0
            print(f"   ❌ LLM error: {e}")
        
        overall_time = search_time + context_time + llm_time
        
        print(f"   Search: {search_time:.2f}s | Context: {context_time:.2f}s | LLM: {llm_time:.2f}s | Total: {overall_time:.2f}s")
        
        total_search += search_time
        total_context += context_time
        total_llm += llm_time
        total_overall += overall_time
    
    # Averages
    n = len(test_statements)
    print(f"\n📊 AVERAGE TIMING (across {n} statements)")
    print(f"   Search: {total_search/n:.2f}s")
    print(f"   Context: {total_context/n:.2f}s")
    print(f"   LLM: {total_llm/n:.2f}s")
    print(f"   Total: {total_overall/n:.2f}s")

if __name__ == "__main__":
    analyze_timing()
    test_multiple_statements() 