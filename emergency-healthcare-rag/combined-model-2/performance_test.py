#!/usr/bin/env python3
"""
Performance test for combined-model-2
Measures timing and accuracy
"""

import sys
import os
import time
from pathlib import Path

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from model import predict

def performance_test():
    """Test performance with timing"""
    
    test_statements = [
        "Euglycemic diabetic ketoacidosis is characterized by blood glucose less than 250 mg/dL with metabolic acidosis.",
        "Contraindications for bronchoscopy include severe baseline hypoxia, recent myocardial infarction, and severe bleeding disorders.",
        "Ischemic colitis following abdominal aortic aneurysm repair occurs due to interrupted perfusion of the inferior mesenteric artery.",
        "The Gurd and Wilson criteria for fat embolism syndrome require either 2 major criteria or at least 1 major criterion plus 4 minor criteria.",
        "In euglycemic diabetic ketoacidosis management, dextrose 5% should be added to initial fluid resuscitation since glucose levels are less than 250 mg/dL."
    ]
    
    print("⏱️ Performance Test for Combined Model 2...\n")
    
    total_time = 0
    results = []
    
    for i, statement in enumerate(test_statements, 1):
        print(f"Test {i}: {statement[:60]}...")
        
        start_time = time.time()
        try:
            truth, topic = predict(statement)
            end_time = time.time()
            inference_time = end_time - start_time
            
            results.append({
                'statement': statement,
                'truth': truth,
                'topic': topic,
                'time': inference_time
            })
            
            print(f"  Result: Truth={truth}, Topic={topic}")
            print(f"  Time: {inference_time:.2f}s")
            total_time += inference_time
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
        print()
    
    # Print summary
    print("📊 Performance Summary:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per statement: {total_time/len(test_statements):.2f}s")
    print(f"Statements per second: {len(test_statements)/total_time:.2f}")
    
    # Check if within 5-second limit
    if total_time < 5.0:
        print("✅ Within 5-second competition limit!")
    else:
        print("⚠️ Exceeds 5-second competition limit")

if __name__ == "__main__":
    performance_test() 