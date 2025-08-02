#!/usr/bin/env python3
"""
Debug script to check if correct topics are found by semantic search
"""

import sys
import json
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from search import get_top_k_topics_with_context

def load_topics_mapping():
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def debug_search_results():
    """
    Check if correct topics are found in search results
    """
    topics_mapping = load_topics_mapping()
    
    # Test statements from evaluation
    test_cases = [
        {
            'statement': 'Coronary heart disease affects approximately 15.5 million people in the United States.',
            'expected_topic': 4,
            'expected_topic_name': 'Acute Coronary Syndrome'
        },
        {
            'statement': 'The ureter receives segmental blood supply from the renal arteries proximally, common iliac and gonadal arteries in the middle segment, and external iliac artery branches distally.',
            'expected_topic': 91,
            'expected_topic_name': 'CT Other'
        },
        {
            'statement': 'Intravesicular catheter pressure measurement using a Foley catheter provides accurate assessment of bladder pressure.',
            'expected_topic': 41,
            'expected_topic_name': 'Central Venous Pressure'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n=== Test Case {i} ===")
        print(f"Statement: {case['statement']}")
        print(f"Expected Topic: {case['expected_topic']} ({case['expected_topic_name']})")
        
        try:
            # Get search results
            candidates = get_top_k_topics_with_context(case['statement'], k=8)
            
            print(f"\nTop 8 Search Results:")
            found_correct = False
            for j, candidate in enumerate(candidates, 1):
                topic_id = candidate['topic_id']
                topic_name = [name for name, tid in topics_mapping.items() if tid == topic_id][0]
                score = candidate['score']
                
                if topic_id == case['expected_topic']:
                    found_correct = True
                    print(f"  {j}. ✅ {topic_id}: {topic_name} (score: {score:.4f})")
                else:
                    print(f"  {j}. ❌ {topic_id}: {topic_name} (score: {score:.4f})")
            
            if found_correct:
                print("✅ Correct topic found in search results!")
            else:
                print("❌ Correct topic NOT found in search results!")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_search_results() 