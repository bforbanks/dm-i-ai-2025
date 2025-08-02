#!/usr/bin/env python3
"""
Test script for combined-model-2
Tests various medical statements to see performance
"""

import sys
import os
from pathlib import Path

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from model import predict

def test_examples():
    """Test various medical statements"""
    
    test_cases = [
        {
            "statement": "Euglycemic diabetic ketoacidosis is characterized by blood glucose less than 250 mg/dL with metabolic acidosis.",
            "expected_topic": 30,  # Diabetic Ketoacidosis
            "expected_truth": 1
        },
        {
            "statement": "Contraindications for bronchoscopy include severe baseline hypoxia, recent myocardial infarction, and severe bleeding disorders.",
            "expected_topic": 88,  # Bronchoscopy with BAL
            "expected_truth": 1
        },
        {
            "statement": "Ischemic colitis following abdominal aortic aneurysm repair occurs due to interrupted perfusion of the inferior mesenteric artery.",
            "expected_topic": 70,  # Ruptured AAA
            "expected_truth": 1
        },
        {
            "statement": "The Gurd and Wilson criteria for fat embolism syndrome require either 2 major criteria or at least 1 major criterion plus 4 minor criteria.",
            "expected_topic": 33,  # Embolism
            "expected_truth": 1
        },
        {
            "statement": "In euglycemic diabetic ketoacidosis management, dextrose 5% should be added to initial fluid resuscitation since glucose levels are less than 250 mg/dL.",
            "expected_topic": 30,  # Diabetic Ketoacidosis
            "expected_truth": 1
        },
        {
            "statement": "Acute myocardial infarction is characterized by ST-segment elevation on ECG and elevated troponin levels.",
            "expected_topic": 7,  # Acute Myocardial Infarction (STEMI_NSTEMI)
            "expected_truth": 1
        },
        {
            "statement": "Pulmonary embolism is diagnosed using CT angiography and D-dimer testing.",
            "expected_topic": 63,  # Pulmonary Embolism
            "expected_truth": 1
        },
        {
            "statement": "Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection.",
            "expected_topic": 72,  # Sepsis_Septic Shock
            "expected_truth": 1
        },
        {
            "statement": "Acute respiratory distress syndrome requires mechanical ventilation with low tidal volumes.",
            "expected_topic": 8,  # Acute Respiratory Distress Syndrome
            "expected_truth": 1
        },
        {
            "statement": "Cardiac arrest requires immediate CPR and defibrillation for shockable rhythms.",
            "expected_topic": 22,  # Cardiac Arrest
            "expected_truth": 1
        }
    ]
    
    print("🧪 Testing Combined Model 2 with various examples...\n")
    
    correct_truth = 0
    correct_topic = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        statement = test_case["statement"]
        expected_topic = test_case["expected_topic"]
        expected_truth = test_case["expected_truth"]
        
        print(f"Test {i}:")
        print(f"Statement: {statement}")
        print(f"Expected: Truth={expected_truth}, Topic={expected_topic}")
        
        try:
            truth, topic = predict(statement)
            print(f"Predicted: Truth={truth}, Topic={topic}")
            
            # Check accuracy
            if truth == expected_truth:
                correct_truth += 1
                print("✅ Truth correct")
            else:
                print("❌ Truth incorrect")
                
            if topic == expected_topic:
                correct_topic += 1
                print("✅ Topic correct")
            else:
                print("❌ Topic incorrect")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print("-" * 80)
    
    # Print summary
    print(f"\n📊 Results Summary:")
    print(f"Truth Accuracy: {correct_truth}/{total} ({correct_truth/total*100:.1f}%)")
    print(f"Topic Accuracy: {correct_topic}/{total} ({correct_topic/total*100:.1f}%)")
    print(f"Overall Accuracy: {(correct_truth + correct_topic)/(total*2)*100:.1f}%")

if __name__ == "__main__":
    test_examples() 