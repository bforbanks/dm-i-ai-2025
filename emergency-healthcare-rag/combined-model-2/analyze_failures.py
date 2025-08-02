#!/usr/bin/env python3
"""
Analyze specific failure cases to understand why the LLM makes wrong decisions
"""

import sys
import os
import json
import ollama
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from llm import classify_truth_and_topic_combined, generate_classification_prompt
from search import get_top_k_topics_with_context, get_targeted_context_for_topic
from config import get_llm_model, get_model_info

def analyze_failure_case(statement: str, expected_truth: int, expected_topic: int):
    """Analyze a specific failure case in detail"""
    
    print(f"🔍 Analyzing Failure Case")
    print("=" * 80)
    print(f"Statement: {statement}")
    print(f"Expected: Truth={expected_truth}, Topic={expected_topic}")
    print()
    
    # Get search results
    candidate_topics = get_top_k_topics_with_context(statement, k=8)
    print("📋 Search Results (Top 8):")
    for i, topic in enumerate(candidate_topics, 1):
        is_expected = topic['topic_id'] == expected_topic
        marker = "🎯" if is_expected else "  "
        print(f"{marker} {i}. {topic['topic_id']:3d}: {topic['topic_name']}")
    print()
    
    # Test LLM with detailed prompt
    model = get_llm_model()
    model_info = get_model_info(model)
    print(f"🤖 Testing with model: {model_info['name']}")
    print()
    
    # Generate prompt using centralized function
    prompt = generate_classification_prompt(statement, candidate_topics, "")

    try:
        print("📤 Sending prompt to LLM...")
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        response_text = response['message']['content'].strip()
        print(f"📥 LLM Response: '{response_text}'")
        print()
        
        # Parse response
        import re
        numbers = re.findall(r'\d+', response_text)
        if len(numbers) >= 2:
            predicted_topic = int(numbers[0])
            predicted_truth = int(numbers[1])
            
            print(f"🔧 Parsed Result: Topic={predicted_topic}, Truth={predicted_truth}")
            print()
            
            # Analyze the decision
            topic_correct = predicted_topic == expected_topic
            truth_correct = predicted_truth == expected_truth
            
            print("📊 Analysis:")
            print(f"  Topic: {'✅' if topic_correct else '❌'} Predicted {predicted_topic}, Expected {expected_topic}")
            print(f"  Truth: {'✅' if truth_correct else '❌'} Predicted {predicted_truth}, Expected {expected_truth}")
            
            if not truth_correct:
                print(f"\n🤔 Truth Analysis:")
                print(f"  - Expected: {expected_truth} ({'TRUE' if expected_truth else 'FALSE'})")
                print(f"  - Predicted: {predicted_truth} ({'TRUE' if predicted_truth else 'FALSE'})")
                print(f"  - The LLM thinks this statement is {'TRUE' if predicted_truth else 'FALSE'}")
                print(f"  - But it should be {'TRUE' if expected_truth else 'FALSE'}")
            
            if not topic_correct:
                print(f"\n🤔 Topic Analysis:")
                print(f"  - Expected topic {expected_topic} was found in search results")
                print(f"  - But LLM chose topic {predicted_topic}")
                print(f"  - This suggests the LLM is not following the search ranking")
        
        else:
            print("❌ Could not parse response")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def load_and_analyze_failures():
    """Load evaluation results and analyze all failures"""
    
    # Load evaluation results
    results_file = "evaluation_results.json"
    if not os.path.exists(results_file):
        print(f"❌ {results_file} not found. Run evaluate.py first.")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    detailed_results = data.get('detailed_results', [])
    if not detailed_results:
        print("❌ No detailed results found in evaluation file.")
        return
    
    # Find all failures
    failures = []
    for result in detailed_results:
        if not result['truth_correct'] or not result['topic_correct']:
            failures.append(result)
    
    print(f"🔍 Found {len(failures)} failure cases out of {len(detailed_results)} total samples")
    print("=" * 80)
    
    # Analyze each failure
    for i, failure in enumerate(failures, 1):
        print(f"\n📝 Failure Case {i}/{len(failures)}")
        print("-" * 60)
        analyze_failure_case(
            failure['statement'], 
            failure['expected_truth'], 
            failure['expected_topic']
        )
        print()

def test_specific_failures():
    """Test the specific failure cases from the evaluation"""
    
    failure_cases = [
        {
            "statement": "The ureter receives segmental blood supply from the renal artery, abdominal aorta, and common iliac artery.",
            "expected_truth": 0,  # This should be FALSE
            "expected_topic": 91,
            "description": "Ureter blood supply - should be FALSE"
        },
        {
            "statement": "Intravesicular catheter pressure measurement using a Foley catheter is a standard diagnostic procedure for bladder outlet obstruction.",
            "expected_truth": 0,  # This should be FALSE  
            "expected_topic": 41,
            "description": "Foley catheter pressure - should be FALSE"
        }
    ]
    
    print("🧪 Analyzing Specific Failure Cases")
    print("=" * 80)
    
    for i, case in enumerate(failure_cases, 1):
        print(f"\n📝 Test Case {i}: {case['description']}")
        print("-" * 60)
        analyze_failure_case(case["statement"], case["expected_truth"], case["expected_topic"])
        print()

def main():
    """Main function"""
    print("🔍 Failure Analysis Tool")
    print("=" * 80)
    
    # Check if evaluation results exist
    if os.path.exists("evaluation_results.json"):
        print("📊 Loading and analyzing all failures from evaluation results...")
        load_and_analyze_failures()
    else:
        print("📊 No evaluation results found, running test cases...")
        test_specific_failures()
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main() 