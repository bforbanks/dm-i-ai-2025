#!/usr/bin/env python3
"""
Warm-up script for the medical RAG model
Preloads all models and embeddings to avoid cold start delays
"""

import time
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model import predict, ACTIVE_MODEL

def warm_up_models():
    """Preload all models and embeddings"""
    print("🔥 Warming up medical RAG models...")
    print(f"📦 Active model: {ACTIVE_MODEL}")
    
    try:
        # Test prediction to warm up all components
        test_statement = "Euglycemic diabetic ketoacidosis is characterized by blood glucose less than 250 mg/dL with metabolic acidosis."
        print("🧪 Running test prediction to warm up models...")
        
        start_time = time.time()
        truth, topic = predict(test_statement)
        warmup_time = time.time() - start_time
        
        print(f"✅ Warm-up complete!")
        print(f"📊 Test prediction: truth={truth}, topic={topic}")
        print(f"⏱️  Warm-up time: {warmup_time:.2f}s")
        print()
        print("🚀 Models are ready! You can now start the API server.")
        print("   Run: python api.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Warm-up failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    warm_up_models() 