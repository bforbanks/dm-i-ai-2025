#!/usr/bin/env python3
"""
Setup script for separated-models-1
Generates embeddings and tests the model
"""

import sys
import os
from pathlib import Path

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from search import create_embeddings, load_embeddings
from model import predict

def main():
    print("🚀 Setting up Separated Models 1...")
    
    if not os.path.exists("data/condensed_topics"):
        print("❌ Error: data/condensed_topics directory not found!")
        print("Please run the topics processing scripts first:")
        print("  python topics_processing/create_condensed_topics.py")
        print("  python topics_processing/clean_condensed_topics.py")
        return
    
    print("📚 Generating embeddings and BM25 index...")
    create_embeddings()
    
    print("🧪 Testing the model...")
    test_statement = "Euglycemic diabetic ketoacidosis is characterized by blood glucose less than 250 mg/dL with metabolic acidosis."
    
    try:
        truth, topic = predict(test_statement)
        print(f"✅ Test successful!")
        print(f"Statement: {test_statement}")
        print(f"Truth: {truth}")
        print(f"Topic: {topic}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("🎉 Separated Models 1 setup complete!")

if __name__ == "__main__":
    main() 