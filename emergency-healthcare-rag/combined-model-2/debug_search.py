#!/usr/bin/env python3
"""
Debug script to analyze search results and understand topic selection failures
"""

import sys
import os
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from search import get_top_k_topics_with_context, get_targeted_context_for_topic

def analyze_search_results(statement: str, expected_topic: int, k: int = 10):
    """Analyze search results for a specific statement"""
    
    print(f"🔍 Analyzing search for statement: {statement[:100]}...")
    print(f"Expected topic: {expected_topic}")
    print("=" * 80)
    
    # Get search results
    candidate_topics = get_top_k_topics_with_context(statement, k=k)
    
    print(f"📋 Top {len(candidate_topics)} search results:")
    print("-" * 80)
    
    found_expected = False
    for i, topic in enumerate(candidate_topics, 1):
        is_expected = topic['topic_id'] == expected_topic
        if is_expected:
            found_expected = True
            print(f"🎯 {i:2d}. {topic['topic_id']:3d}: {topic['topic_name']} (EXPECTED)")
        else:
            print(f"    {i:2d}. {topic['topic_id']:3d}: {topic['topic_name']}")
        
        # Show scores if available
        if 'bm25_score' in topic:
            print(f"        BM25: {topic['bm25_score']:.4f}")
        if 'vector_score' in topic:
            print(f"        Vector: {topic['vector_score']:.4f}")
        if 'combined_score' in topic:
            print(f"        Combined: {topic['combined_score']:.4f}")
        print()
    
    if not found_expected:
        print(f"❌ Expected topic {expected_topic} NOT found in top {k} results!")
        
        # Check if it exists in the full dataset
        try:
            with open('data/topics.json', 'r') as f:
                topics_data = json.load(f)
            
            if str(expected_topic) in topics_data:
                expected_topic_name = topics_data[str(expected_topic)]
                print(f"ℹ️  Expected topic {expected_topic} exists in dataset: {expected_topic_name}")
            else:
                print(f"⚠️  Expected topic {expected_topic} not found in topics.json")
        except Exception as e:
            print(f"⚠️  Could not check topics.json: {e}")
    else:
        print(f"✅ Expected topic {expected_topic} found in search results!")
    
    return candidate_topics

def test_multiple_statements():
    """Test search with multiple statements from the evaluation"""
    
    test_cases = [
        {
            "statement": "Coronary heart disease affects approximately 15.5 million people in the United States.",
            "expected_topic": 4,
            "expected_truth": 1
        },
        {
            "statement": "Neonatal testicular torsion typically occurs as extravaginal torsion.",
            "expected_topic": 78,
            "expected_truth": 1
        },
        {
            "statement": "The Gurd and Wilson criteria for fat embolism syndrome require at least one major and four minor criteria.",
            "expected_topic": 33,
            "expected_truth": 1
        },
        {
            "statement": "The ureter receives segmental blood supply from the renal artery, abdominal aorta, and common iliac artery.",
            "expected_topic": 91,
            "expected_truth": 0
        },
        {
            "statement": "Intravesicular catheter pressure measurement using a Foley catheter is a standard diagnostic procedure for bladder outlet obstruction.",
            "expected_topic": 41,
            "expected_truth": 0
        }
    ]
    
    print("🧪 Testing Search with Multiple Statements")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}")
        print("-" * 40)
        analyze_search_results(case["statement"], case["expected_topic"], k=8)
        print()

def analyze_context_quality(statement: str, topic_id: int):
    """Analyze the quality of context retrieved for a specific topic"""
    
    print(f"📖 Analyzing context quality for topic {topic_id}")
    print("=" * 60)
    
    # Get context
    context = get_targeted_context_for_topic(statement, topic_id, max_chars=2000)
    
    print(f"Context length: {len(context)} characters")
    print(f"Context preview:")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-" * 40)
    
    # Analyze context relevance
    statement_words = set(statement.lower().split())
    context_words = set(context.lower().split())
    
    # Find overlapping medical terms
    medical_terms = [
        'disease', 'syndrome', 'artery', 'vein', 'heart', 'lung', 'kidney', 'liver',
        'blood', 'pressure', 'pain', 'treatment', 'diagnosis', 'symptoms', 'causes',
        'risk', 'factors', 'complications', 'management', 'therapy', 'medication'
    ]
    
    overlapping_medical = [term for term in medical_terms if term in statement_words and term in context_words]
    
    print(f"📊 Context Analysis:")
    print(f"  - Statement words: {len(statement_words)}")
    print(f"  - Context words: {len(context_words)}")
    print(f"  - Overlapping medical terms: {overlapping_medical}")
    print(f"  - Relevance score: {len(overlapping_medical)}/{len([t for t in medical_terms if t in statement_words])}")

def main():
    """Main function"""
    print("🔍 Search Debug Tool")
    print("=" * 80)
    
    # Test with multiple statements
    test_multiple_statements()
    
    # Test context quality for a specific case
    print("\n" + "=" * 80)
    statement = "Coronary heart disease affects approximately 15.5 million people in the United States."
    analyze_context_quality(statement, 4)
    
    print("\n✅ Search debug complete!")

if __name__ == "__main__":
    main() 