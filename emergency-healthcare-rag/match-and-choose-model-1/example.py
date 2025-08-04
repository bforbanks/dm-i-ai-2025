#!/usr/bin/env python3
"""
Simple example usage of match-and-choose-model-1
Demonstrates the threshold-based decision making
"""

from model import predict_with_details
from config import set_threshold, set_llm_model

def main():
    """Simple example of using the match-and-choose model"""
    
    # Example statement that creates a tie (gap = 0)
    statement = "Intravesicular catheter pressure measurement using a Foley catheter requires injection of 50 cc of sterile saline."
    
    print("🎯 MATCH-AND-CHOOSE MODEL EXAMPLE")
    print("=" * 60)
    print(f"Statement: {statement}")
    print()
    
    # Set to a smaller model for local testing if needed
    # set_llm_model("gemma3n:e4b")  # Uncomment if you have this model
    
    # Test with threshold = 0 (should use LLM for tied cases)
    print("📊 THRESHOLD = 0 (LLM decides when scores are tied)")
    print("-" * 50)
    result = predict_with_details(statement, threshold=0)
    
    print(f"Prediction: Truth = {result['prediction']['truth_value']}, Topic = {result['prediction']['topic_id']}")
    print(f"Approach used: {result['decision_info']['approach_used']}")
    print(f"Score gap: {result['decision_info']['gap']:.3f}")
    print(f"Top topics: {[f\"Topic {t['topic_id']} ({t['score']:.2f})\" for t in result['decision_info']['top_topics'][:2]]}")
    print()
    
    # Compare with threshold = NA (always separated)
    print("📊 THRESHOLD = NA (always use topic model + truth LLM)")
    print("-" * 50)
    result_na = predict_with_details(statement, threshold='NA')
    
    print(f"Prediction: Truth = {result_na['prediction']['truth_value']}, Topic = {result_na['prediction']['topic_id']}")
    print(f"Approach used: {result_na['decision_info']['approach_used']}")
    print()
    
    print("✅ The match-and-choose model successfully demonstrates:")
    print("   • Threshold-based decision making")
    print("   • Automatic approach selection (separated vs combined)")
    print("   • Gap analysis for uncertainty detection")
    print()
    print("💡 Next steps:")
    print("   • Test with actual LLM models when available")
    print("   • Run evaluation on validation set")
    print("   • Optimize threshold based on quantitative results")

if __name__ == "__main__":
    main()