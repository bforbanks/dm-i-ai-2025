#!/usr/bin/env python3
"""
Example usage of separated-models-2 RAG system
Demonstrates BM25-only search with LLM
"""

from model import create_rag_model
from config import list_available_models, set_llm_model

def main():
    """Example usage of the RAG system"""
    print("🚑 EMERGENCY HEALTHCARE RAG - SEPARATED MODELS 2")
    print("=" * 60)
    print("BM25-only search with optimized chunking")
    print("=" * 60)
    
    # Show available models
    print("\n📋 Available LLM Models:")
    list_available_models()
    
    # Create RAG model
    print("\n🤖 Creating RAG model...")
    rag_model = create_rag_model()
    
    # Show model configuration
    model_info = rag_model.get_model_info()
    print(f"Model Configuration:")
    print(f"  LLM: {model_info['llm_model']}")
    print(f"  Search: {model_info['search_method']}")
    print(f"  Data: {model_info['data_source']}")
    
    # Example statements
    example_statements = [
        "Chest pain that radiates to the left arm is always a sign of myocardial infarction.",
        "A patient with acute appendicitis typically presents with right lower quadrant pain.",
        "Pulmonary embolism can be diagnosed with a normal D-dimer test.",
        "Diabetic ketoacidosis is characterized by high blood glucose and low ketone levels.",
        "A patient with sepsis will always have a fever above 38°C."
    ]
    
    print(f"\n🔍 Processing {len(example_statements)} example statements...")
    print("-" * 60)
    
    for i, statement in enumerate(example_statements, 1):
        print(f"\n📝 Statement {i}: {statement}")
        print("-" * 40)
        
        # Process through RAG pipeline
        result = rag_model.process_statement(statement)
        
        print(f"Topic ID: {result['topic_id']}")
        print(f"Response: {result['response']} (1=TRUE, 0=FALSE)")
        print(f"Model: {result['model_used']}")
        
        # Show a snippet of context
        context_snippet = result['context'][:200] + "..." if len(result['context']) > 200 else result['context']
        print(f"Context snippet: {context_snippet}")
    
    print("\n" + "=" * 60)
    print("✅ Example completed!")
    print("=" * 60)

if __name__ == "__main__":
    main() 