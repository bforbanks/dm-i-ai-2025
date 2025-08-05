#!/usr/bin/env python3
"""
Example usage of the hierarchical topic model optimization

This script demonstrates how to run the 4-phase optimization
and analyze the results.
"""

import json
from pathlib import Path
from optimize_topic_model import run_optimization, OptimizationConfig

def main():
    """Run the optimization and analyze results"""
    
    print("🚀 Starting Topic Model Optimization Example")
    print("=" * 50)
    
    # Option 1: Run full optimization (2-4 hours)
    print("\n📊 Option 1: Full 4-Phase Optimization")
    print("   This will run all phases and save results")
    print("   Expected runtime: 2-4 hours")
    
    # Uncomment to run full optimization:
    # run_optimization()
    
    # Option 2: Custom configuration
    print("\n📊 Option 2: Custom Configuration")
    print("   You can customize the optimization parameters:")
    
    config = OptimizationConfig(
        # Phase 1: BM25 optimization
        bm25_chunk_sizes=[96, 112, 128, 144, 160],  # Reduced for faster testing
        bm25_overlap_ratios=[0.05, 0.1, 0.15, 0.2],  # Reduced for faster testing
        
        # Phase 2: Embedding optimization
        embedding_models=[
            "sentence-transformers/all-MiniLM-L6-v2",  # Fast testing
            "sentence-transformers/all-mpnet-base-v2",  # Best performance
        ],
        embedding_chunk_sizes=[112, 128, 144],
        embedding_overlap_ratios=[0.1, 0.15, 0.2],
        
        # Phase 3: Combination parameters
        top_bm25_configs=3,
        top_embedding_configs=3,
        
        # Phase 4: Zoom parameters
        zoom_enabled=True,
        zoom_radius=1,  # Smaller radius for faster testing
        
        # General parameters
        max_samples=100,  # Reduced for faster testing
    )
    
    print("   Custom config created with reduced parameters for faster testing")
    print("   To use this config, modify optimize_topic_model.py to use it")
    
    # Option 3: Analyze existing results
    print("\n📊 Option 3: Analyze Existing Results")
    print("   If you have existing optimization results, you can analyze them:")
    
    # Example of how to load and analyze results
    def analyze_results(filename: str):
        """Analyze optimization results"""
        if Path(filename).exists():
            with open(filename, 'r') as f:
                results = json.load(f)
            
            print(f"\n📈 Results from {filename}:")
            print(f"   Total configurations tested: {len(results)}")
            
            if results:
                best_result = results[0]
                print(f"   Best accuracy: {best_result['metrics']['accuracy']:.3f}")
                print(f"   Best configuration: {best_result['config']}")
                
                # Show top 5 results
                print(f"\n🏆 Top 5 configurations:")
                for i, result in enumerate(results[:5], 1):
                    metrics = result['metrics']
                    config = result['config']
                    print(f"   {i}. Accuracy: {metrics['accuracy']:.3f}, Config: {config}")
    
    # Check for existing results
    result_files = list(Path(".").glob("optimization_results_*.json"))
    if result_files:
        print(f"   Found {len(result_files)} result files:")
        for file in result_files:
            print(f"     - {file.name}")
            analyze_results(str(file))
    else:
        print("   No existing result files found")
        print("   Run the optimization first to generate results")
    
    print("\n✅ Example completed!")
    print("\nTo run the full optimization:")
    print("   cd emergency-healthcare-rag/match-and-choose-model-1/")
    print("   python optimize_topic_model.py")

if __name__ == "__main__":
    main()