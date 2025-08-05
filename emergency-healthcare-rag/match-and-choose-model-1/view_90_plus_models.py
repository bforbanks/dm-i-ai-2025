#!/usr/bin/env python3
"""
View all models that scored 90% or above
"""

import json
from pathlib import Path
from datetime import datetime

def view_90_plus_models():
    """Display all models that scored 90% or above"""
    filename = "models_90_plus.json"
    
    # Use script directory for path anchoring
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / filename
    
    if not file_path.exists():
        print(f"❌ No 90%+ models file found: {filename}")
        print("   Run the optimization script first to generate results.")
        return
    
    try:
        with open(file_path, 'r') as f:
            models = json.load(f)
        
        if not models:
            print("No models with 90%+ accuracy found yet.")
            return
        
        print("🎯 MODELS WITH 90%+ ACCURACY")
        print("=" * 80)
        print(f"Total 90%+ models found: {len(models)}")
        print()
        
        # Sort by accuracy (highest first)
        models.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        
        for i, model in enumerate(models, 1):
            accuracy = model.get('accuracy', 0)
            timestamp = model.get('timestamp', 0)
            config = model.get('config', {})
            metrics = model.get('metrics', {})
            
            print(f"{i}. Accuracy: {accuracy:.3f} | Found: {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}")
            print(f"   Avg Rank: {metrics.get('avg_rank', 0):.2f} | Top-3: {metrics.get('top3_accuracy', 0):.3f}")
            
            # Display configuration details
            if 'model_name' in config:
                print(f"   Model: {config['model_name']} | Strategy: {config.get('strategy', 'unknown')}")
                print(f"   Chunk: {config.get('chunk_size')}, Overlap: {config.get('overlap')}")
            elif 'fusion_strategy' in config:
                print(f"   Strategy: {config['fusion_strategy']}")
                bm25_cfg = config.get('bm25_config', {})
                semantic_cfg = config.get('semantic_config', {})
                print(f"   BM25: chunk={bm25_cfg.get('chunk_size')}, overlap={bm25_cfg.get('overlap')}")
                print(f"   Semantic: {semantic_cfg.get('model_name')}, chunk={semantic_cfg.get('chunk_size')}")
            else:
                print(f"   Strategy: {config.get('strategy', 'unknown')}")
                print(f"   Chunk: {config.get('chunk_size')}, Overlap: {config.get('overlap')}")
            
            print()
        
        print(f"💾 All 90%+ models saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error reading 90%+ models file: {e}")

if __name__ == "__main__":
    view_90_plus_models() 