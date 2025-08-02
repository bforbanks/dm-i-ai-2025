#!/usr/bin/env python3
"""
Quick speed test for UCloud deployment
Run this to verify your model meets the <5 second constraint
"""
import time
import sys
import os
sys.path.append('model-1')

from model import predict

# Test statements of varying complexity
test_statements = [
    "Heart attack requires immediate medical attention",
    "Pneumonia is treated with antibiotics in most cases",
    "CT scans use ionizing radiation to create detailed images",
    "Stroke symptoms include sudden weakness and speech difficulties",
    "Sepsis can lead to organ failure if untreated"
]

def test_speed():
    print("🚀 SPEED TEST FOR UCLOUD DEPLOYMENT")
    print("=" * 50)
    
    total_time = 0
    for i, statement in enumerate(test_statements):
        print(f"\n📝 Test {i+1}: {statement[:50]}...")
        
        start_time = time.time()
        try:
            truth, topic = predict(statement)
            end_time = time.time()
            
            duration = end_time - start_time
            total_time += duration
            
            status = "✅ PASS" if duration < 5.0 else "❌ FAIL"
            print(f"   Result: truth={truth}, topic={topic}")
            print(f"   Time: {duration:.2f}s {status}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return False
    
    avg_time = total_time / len(test_statements)
    print(f"\n🏁 FINAL RESULTS:")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time: {avg_time:.2f}s")
    
    if avg_time < 5.0:
        print(f"   🎉 SUCCESS! Your model meets the speed requirement!")
        return True
    else:
        print(f"   ⚠️  TOO SLOW! Need to optimize further.")
        print(f"   Suggestions:")
        print(f"   - Use smaller Ollama model (gemma3:3b)")
        print(f"   - Reduce max_chars in context")
        print(f"   - Set TOPIC_CANDIDATES_K = 1")
        return False

if __name__ == "__main__":
    test_speed()