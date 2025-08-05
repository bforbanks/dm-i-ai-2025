#!/usr/bin/env python3
"""
View the current top 5 models from the optimization
"""

import json
from pathlib import Path
from datetime import datetime

def view_top_models():
    """Display the current top 5 models"""
    filename = "top_5_models.json"
    
    # Use script directory for path anchoring
    script_dir = Path(__file__).resolve().parent
    file_path = script_dir / filename
    
    if not file_path.exists():
        print(f"❌ No top models file found: {filename}")
        print("   Run the optimization script first to generate results.")
        return
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        last_updated = data.get('last_updated', 0)
        total_tested = data.get('total_configurations_tested', 0)
        top_models = data.get('top_5_models', [])
        
        print("🏆 CURRENT TOP 5 MODELS")
        print("=" * 60)
        print(f"Last updated: {datetime.fromtimestamp(last_updated).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total configurations tested: {total_tested}")
        print()
        
        if not top_models:
            print("No models found in results.")
            return
        
        for i, model in enumerate(top_models, 1):
            rank = model.get('rank', i)
            accuracy = model.get('accuracy', 0)
            avg_rank = model.get('avg_rank', 0)
            top3_acc = model.get('top3_accuracy', 0)
            config = model.get('config', {})
            
            print(f"{rank}. Accuracy: {accuracy:.3f} | Avg Rank: {avg_rank:.2f} | Top-3: {top3_acc:.3f}")
            
            # Display configuration details
            if 'model_name' in config:
                print(f"   Model: {config['model_name']}")
                print(f"   Strategy: {config.get('strategy', 'unknown')}")
                print(f"   Chunk Size: {config.get('chunk_size')}")
                print(f"   Overlap: {config.get('overlap')}")
            elif 'fusion_strategy' in config:
                print(f"   Strategy: {config['fusion_strategy']}")
                bm25_cfg = config.get('bm25_config', {})
                semantic_cfg = config.get('semantic_config', {})
                print(f"   BM25: chunk={bm25_cfg.get('chunk_size')}, overlap={bm25_cfg.get('overlap')}")
                print(f"   Semantic: {semantic_cfg.get('model_name')}, chunk={semantic_cfg.get('chunk_size')}")
            else:
                print(f"   Strategy: {config.get('strategy', 'unknown')}")
                print(f"   Chunk Size: {config.get('chunk_size')}")
                print(f"   Overlap: {config.get('overlap')}")
            
            print()
        
        print("💡 To see live updates, run: python monitor_results.py")
        
    except Exception as e:
        print(f"❌ Error reading top models file: {e}")

if __name__ == "__main__":
    view_top_models() 