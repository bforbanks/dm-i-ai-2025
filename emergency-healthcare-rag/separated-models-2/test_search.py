#!/usr/bin/env python3
"""
Test script for separated-models-2 search functionality
Tests BM25 search without requiring LLM
"""

from search import bm25_search, get_best_topic, get_rich_context_for_statement

def test_search_functionality():
    """Test the search functionality with example statements"""
    print("🔍 TESTING SEPARATED MODELS 2 SEARCH FUNCTIONALITY")
    print("=" * 60)
    
    # Test statements
    test_statements = [
        "Chest pain that radiates to the left arm is always a sign of myocardial infarction.",
        "A patient with acute appendicitis typically presents with right lower quadrant pain.",
        "Pulmonary embolism can be diagnosed with a normal D-dimer test.",
        "Diabetic ketoacidosis is characterized by high blood glucose and low ketone levels.",
        "A patient with sepsis will always have a fever above 38°C."
    ]
    
    for i, statement in enumerate(test_statements, 1):
        print(f"\n📝 Test Statement {i}: {statement}")
        print("-" * 50)
        
        # Test topic identification
        topic_id = get_best_topic(statement)
        print(f"Identified Topic ID: {topic_id}")
        
        # Test BM25 search
        search_results = bm25_search(statement, top_k=3)
        print(f"Top 3 search results:")
        for j, result in enumerate(search_results, 1):
            print(f"  {j}. Topic: {result['topic_name']} (ID: {result['topic_id']}) - Score: {result['score']:.3f}")
        
        # Test context retrieval
        context = get_rich_context_for_statement(statement, topic_id, max_chars=300)
        print(f"Context snippet: {context[:200]}...")
    
    print("\n" + "=" * 60)
    print("✅ Search functionality test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_search_functionality() 