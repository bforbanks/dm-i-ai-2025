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
from llm import classify_truth_and_topic_combined
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
    
    # Get context for expected topic
    context = get_targeted_context_for_topic(statement, expected_topic, max_chars=2000)
    print(f"📖 Context for expected topic {expected_topic}:")
    print("-" * 60)
    print(context[:800] + "..." if len(context) > 800 else context)
    print("-" * 60)
    print()
    
    # Test LLM with detailed prompt
    model = get_llm_model()
    model_info = get_model_info(model)
    print(f"🤖 Testing with model: {model_info['name']}")
    print()
    
    candidates_text = "\n".join([
        f"{topic['topic_id']}: {topic['topic_name']}" 
        for topic in candidate_topics
    ])
    
    # Create a more explicit prompt
    prompt = f"""You are a medical expert. Analyze this statement and provide ONLY two numbers: topic_id,truth_value

STATEMENT: {statement}

MEDICAL CONTEXT:
{context}

AVAILABLE TOPICS (choose the most relevant):
{candidates_text}

INSTRUCTIONS:
1. Choose the topic ID (0-114) that best matches the medical content
2. Determine if the statement is TRUE (1) or FALSE (0) based on the context
3. Respond with ONLY: topic_id,truth_value
4. Example: 4,1

IMPORTANT: Be very careful about truth determination. Many medical statements are FALSE even if they sound plausible."""

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
                
                # Check if the context supports the expected truth
                context_lower = context.lower()
                statement_lower = statement.lower()
                
                # Look for contradicting information
                contradicting_terms = []
                if expected_truth == 0:  # Should be false
                    # Look for terms that might contradict the statement
                    if "not" in context_lower or "incorrect" in context_lower or "false" in context_lower:
                        contradicting_terms.append("negation terms found")
                    if "different" in context_lower or "alternative" in context_lower:
                        contradicting_terms.append("alternative information found")
                
                if contradicting_terms:
                    print(f"  - Context analysis: {', '.join(contradicting_terms)}")
                else:
                    print(f"  - Context analysis: No obvious contradicting information found")
            
            if not topic_correct:
                print(f"\n🤔 Topic Analysis:")
                print(f"  - Expected topic {expected_topic} was found in search results")
                print(f"  - But LLM chose topic {predicted_topic}")
                print(f"  - This suggests the LLM is not following the search ranking")
        
        else:
            print("❌ Could not parse response")
            
    except Exception as e:
        print(f"❌ Error: {e}")

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
        print(f"\n📝 Failure Case {i}: {case['description']}")
        print("-" * 60)
        analyze_failure_case(case["statement"], case["expected_truth"], case["expected_topic"])
        print()

def main():
    """Main function"""
    print("🔍 Failure Analysis Tool")
    print("=" * 80)
    
    test_specific_failures()
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main() 