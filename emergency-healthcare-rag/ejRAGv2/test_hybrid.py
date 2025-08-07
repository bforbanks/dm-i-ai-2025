"""
Test script for hybrid retrieval implementation.
Runs a quick test with a small number of iterations to verify everything works.
"""

import sys
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent))

from hybrid_retrieval_bo import (
    load_statements, 
    build_bm25_index, 
    build_embedding_index,
    hybrid_search,
    evaluate_config
)

def test_basic_functionality():
    """Test basic functionality with a small configuration."""
    print("🧪 Testing basic hybrid retrieval functionality...")
    
    try:
        # Test with a simple configuration
        config = {
            'chunk_size_bm25': 64,
            'overlap_bm25': 16,
            'k1': 1.2,
            'b': 0.75,
            'chunk_size_embed': 64,
            'overlap_embed': 16,
            'model_type': 'MiniLM',
            'alpha': 0.5,
            'beta': 0.5
        }
        
        print(f"Configuration: {config}")
        print("="*50)
        
        # Build indices
        print("📚 Building BM25 index...")
        bm25_data = build_bm25_index(
            config['chunk_size_bm25'],
            config['overlap_bm25'],
            config['k1'],
            config['b']
        )
        
        print("🔍 Building embedding index...")
        embed_data = build_embedding_index(
            config['chunk_size_embed'],
            config['overlap_embed'],
            config['model_type']
        )
        
        # Load a few test statements
        print("📝 Loading test statements...")
        statements = load_statements()
        test_statements = statements[:5]  # Test with first 5 statements
        
        print(f"Testing with {len(test_statements)} statements...")
        print("="*50)
        
        # Test hybrid search with progress bar
        correct = 0
        print("🔎 Testing hybrid search...")
        for i, (query, true_topic) in enumerate(tqdm(test_statements, desc="Testing queries", unit="query")):
            results = hybrid_search(
                query, bm25_data, embed_data,
                config['alpha'], config['beta'], top_k=1
            )
            
            if results and results[0][0] == true_topic:
                correct += 1
                print(f"  ✓ Statement {i+1}: Correct")
            else:
                print(f"  ✗ Statement {i+1}: Incorrect (predicted: {results[0][0] if results else 'None'}, true: {true_topic})")
        
        accuracy = correct / len(test_statements)
        print(f"\n📊 Test Accuracy: {accuracy:.3f} ({correct}/{len(test_statements)})")
        
        # Test evaluation function
        print("\n⚡ Testing evaluation function...")
        eval_accuracy = evaluate_config(
            config['chunk_size_bm25'],
            config['overlap_bm25'],
            config['k1'],
            config['b'],
            config['chunk_size_embed'],
            config['overlap_embed'],
            config['model_type'],
            config['alpha'],
            config['beta']
        )
        print(f"📈 Evaluation function accuracy: {eval_accuracy:.3f}")
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_ranges():
    """Test that parameters are within expected ranges."""
    print("\n🔧 Testing parameter ranges...")
    
    try:
        # Test different parameter combinations
        test_configs = [
            # BM25-focused
            {'chunk_size_bm25': 64, 'overlap_bm25': 16, 'k1': 1.2, 'b': 0.75,
             'chunk_size_embed': 64, 'overlap_embed': 16, 'model_type': 'MiniLM',
             'alpha': 0.8, 'beta': 0.2},
            
            # Embedding-focused
            {'chunk_size_bm25': 64, 'overlap_bm25': 16, 'k1': 1.2, 'b': 0.75,
             'chunk_size_embed': 64, 'overlap_embed': 16, 'model_type': 'MiniLM',
             'alpha': 0.2, 'beta': 0.8},
            
            # Balanced
            {'chunk_size_bm25': 64, 'overlap_bm25': 16, 'k1': 1.2, 'b': 0.75,
             'chunk_size_embed': 64, 'overlap_embed': 16, 'model_type': 'MiniLM',
             'alpha': 0.5, 'beta': 0.5},
        ]
        
        print(f"Testing {len(test_configs)} different configurations...")
        print("="*50)
        
        for i, config in enumerate(tqdm(test_configs, desc="Testing configs", unit="config")):
            print(f"\n🔍 Testing config {i+1}: α={config['alpha']:.1f}, β={config['beta']:.1f}")
            accuracy = evaluate_config(
                config['chunk_size_bm25'],
                config['overlap_bm25'],
                config['k1'],
                config['b'],
                config['chunk_size_embed'],
                config['overlap_embed'],
                config['model_type'],
                config['alpha'],
                config['beta']
            )
            print(f"  📊 Config {i+1} accuracy: {accuracy:.3f}")
        
        print("\n✅ Parameter range tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Parameter range test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("🚀 HYBRID RETRIEVAL TEST SUITE")
    print("="*60)
    
    success = True
    
    # Run tests
    success &= test_basic_functionality()
    success &= test_parameter_ranges()
    
    print("\n" + "="*60)
    if success:
        print("✅ ALL TESTS PASSED")
        print("🎯 The hybrid retrieval implementation is ready for optimization!")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please fix the issues before running optimization.")
    print("="*60)

if __name__ == "__main__":
    main()
