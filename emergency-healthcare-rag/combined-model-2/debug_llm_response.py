#!/usr/bin/env python3
"""
Debug script to analyze LLM responses and understand parsing failures
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

def test_llm_parsing():
    """Test LLM response parsing with various formats"""
    
    # Test statement
    statement = "Coronary heart disease affects approximately 15.5 million people in the United States."
    
    print("🔍 Testing LLM Response Parsing")
    print("=" * 60)
    print(f"Statement: {statement}")
    print()
    
    # Get candidates and context
    print("📋 Getting topic candidates...")
    candidate_topics = get_top_k_topics_with_context(statement, k=3)
    print(f"Found {len(candidate_topics)} candidates:")
    for topic in candidate_topics:
        print(f"  {topic['topic_id']}: {topic['topic_name']}")
    print()
    
    # Get context
    print("📖 Getting context...")
    context = get_targeted_context_for_topic(statement, candidate_topics[0]['topic_id'], max_chars=1000)
    print(f"Context length: {len(context)} characters")
    print(f"Context preview: {context[:200]}...")
    print()
    
    # Test LLM call
    model = get_llm_model()
    model_info = get_model_info(model)
    print(f"🤖 Using model: {model_info['name']} ({model_info['description']})")
    print()
    
    try:
        # Call LLM directly to see raw response
        candidates_text = "\n".join([
            f"{topic['topic_id']}: {topic['topic_name']}" 
            for topic in candidate_topics
        ])
        
        prompt = f"""You are a medical expert. Analyze this statement and provide two determinations.

STATEMENT: {statement}

MEDICAL CONTEXT (from the top semantic search result):
{context}

TOPIC CANDIDATES (higher ones are more relevant by semantic search, but search can be wrong):
{candidates_text}

TASKS:
1. Choose the most relevant topic from the candidates above
2. Determine if the statement is TRUE (1) or FALSE (0) based on the context

The chance of a statement being true or false is roughly 50/50. Be skeptical of medical claims unless explicitly confirmed in the context.

Respond with ONLY two numbers separated by a comma: topic_id,truth_value
Examples: 30,1 (topic 30, true) or 45,0 (topic 45, false)"""

        print("📤 Sending prompt to LLM...")
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        response_text = response['message']['content'].strip()
        print(f"📥 Raw LLM Response:")
        print(f"'{response_text}'")
        print()
        
        # Test parsing
        print("🔧 Testing response parsing...")
        
        # Method 1: Simple comma split
        if ',' in response_text:
            parts = response_text.split(',')
            if len(parts) >= 2:
                try:
                    topic_id = int(parts[0].strip())
                    truth_value = int(parts[1].strip())
                    print(f"✅ Method 1 (comma split): topic_id={topic_id}, truth_value={truth_value}")
                except ValueError as e:
                    print(f"❌ Method 1 failed: {e}")
            else:
                print("❌ Method 1 failed: Not enough parts after comma split")
        else:
            print("❌ Method 1 failed: No comma found")
        
        # Method 2: Regex extraction
        import re
        numbers = re.findall(r'\d+', response_text)
        if len(numbers) >= 2:
            try:
                topic_id = int(numbers[0])
                truth_value = int(numbers[1])
                print(f"✅ Method 2 (regex): topic_id={topic_id}, truth_value={truth_value}")
            except ValueError as e:
                print(f"❌ Method 2 failed: {e}")
        else:
            print(f"❌ Method 2 failed: Found {len(numbers)} numbers, need at least 2")
        
        # Method 3: Look for specific patterns
        print(f"🔍 Response analysis:")
        print(f"  - Contains comma: {',' in response_text}")
        print(f"  - Contains newlines: {'\\n' in response_text}")
        print(f"  - Contains explanation: {'explanation' in response_text.lower()}")
        print(f"  - Contains **: {'**' in response_text}")
        print(f"  - Length: {len(response_text)} characters")
        
    except Exception as e:
        print(f"❌ Error calling LLM: {e}")

def test_multiple_formats():
    """Test parsing with various response formats"""
    
    test_responses = [
        "4,1",
        "4, 1",
        "4,1\n",
        "4,1\n\n**Explanation:**\n\n* **Topic Selection:** The statement directly discusses anemia",
        "Topic: 4, Truth: 1",
        "I choose topic 4 and the statement is true (1)",
        "4 and 1",
        "The answer is 4,1",
        "Based on the context, I select topic 4 and determine truth value 1"
    ]
    
    print("\n🧪 Testing Multiple Response Formats")
    print("=" * 60)
    
    for i, response in enumerate(test_responses, 1):
        print(f"\nTest {i}: '{response}'")
        
        # Method 1: Comma split
        if ',' in response:
            parts = response.split(',')
            if len(parts) >= 2:
                try:
                    topic_id = int(parts[0].strip())
                    truth_value = int(parts[1].strip())
                    print(f"  ✅ Comma split: {topic_id}, {truth_value}")
                except ValueError:
                    print(f"  ❌ Comma split failed")
            else:
                print(f"  ❌ Comma split: not enough parts")
        else:
            print(f"  ❌ Comma split: no comma")
        
        # Method 2: Regex
        import re
        numbers = re.findall(r'\d+', response)
        if len(numbers) >= 2:
            try:
                topic_id = int(numbers[0])
                truth_value = int(numbers[1])
                print(f"  ✅ Regex: {topic_id}, {truth_value}")
            except ValueError:
                print(f"  ❌ Regex failed")
        else:
            print(f"  ❌ Regex: found {len(numbers)} numbers")

def main():
    """Main function"""
    print("🐛 LLM Response Debug Tool")
    print("=" * 60)
    
    # Test actual LLM response
    test_llm_parsing()
    
    # Test various response formats
    test_multiple_formats()
    
    print("\n✅ Debug complete!")

if __name__ == "__main__":
    main() 