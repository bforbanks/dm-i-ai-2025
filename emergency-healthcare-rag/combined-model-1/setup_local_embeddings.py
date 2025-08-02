#!/usr/bin/env python3
"""
Setup script for local embeddings and model caching
This downloads the embedding model once and then everything works offline
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('..')

from search import setup_local_model, create_embeddings

def main():
    """Setup local model and generate embeddings"""
    print("🤖 Setting up local embedding model...")
    
    # Ensure we're in the emergency-healthcare-rag directory
    original_dir = os.getcwd()
    if os.path.basename(original_dir) != "emergency-healthcare-rag":
        # We're probably in model-1/, go up one level
        parent_dir = os.path.dirname(original_dir)
        os.chdir(parent_dir)
    
    # Verify data directory exists
    if not os.path.exists("data/condensed_topics"):
        print("❌ Error: data/condensed_topics directory not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Make sure you're running this from the emergency-healthcare-rag directory")
        return 1
    
    try:
        # Step 1: Setup local model (download and cache)
        setup_local_model()
        
        # Step 2: Generate embeddings using local model
        print("\n📊 Generating embeddings with local model...")
        create_embeddings()
        
        print("\n✅ Setup complete! Your model is now fully offline.")
        print("You can now run:")
        print("  python model-1/evaluate_detailed.py 5")
        print("  python example.py")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        return 1
    finally:
        os.chdir(original_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())