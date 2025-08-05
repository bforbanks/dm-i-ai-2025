#!/usr/bin/env python3
"""
Monitor the live optimization results
Run this in a separate terminal to watch progress
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_live_results():
    """Monitor the live results file"""
    filename = "optimization_live_results.json"
    
    print("🔍 Monitoring optimization progress...")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)
    
    last_update = None
    
    while True:
        try:
            if Path(filename).exists():
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Check if there's new data
                current_update = data.get('last_updated', 0)
                if current_update != last_update:
                    last_update = current_update
                    
                    # Display current status
                    status = data.get('status', 'unknown')
                    phase = data.get('phase', 'unknown')
                    total_results = data.get('total_results', 0)
                    
                    print(f"\n📊 Update at {datetime.fromtimestamp(current_update).strftime('%H:%M:%S')}")
                    print(f"   Phase: {phase}")
                    print(f"   Status: {status}")
                    print(f"   Total Results: {total_results}")
                    
                    # Show current config if available
                    current_config = data.get('current_config')
                    if current_config:
                        if 'chunk_size' in current_config:
                            print(f"   Current: chunk_size={current_config.get('chunk_size')}, overlap={current_config.get('overlap')}")
                        elif 'model_name' in current_config:
                            print(f"   Current: {current_config.get('model_name')}, chunk_size={current_config.get('chunk_size')}")
                        elif 'fusion_strategy' in current_config:
                            print(f"   Current: {current_config.get('fusion_strategy')}")
                    
                    # Show top results if available
                    results = data.get('results', [])
                    if results and len(results) > 0:
                        print(f"\n🏆 Top 3 Results:")
                        for i, result in enumerate(results[:3], 1):
                            metrics = result.get('metrics', {})
                            config = result.get('config', {})
                            accuracy = metrics.get('accuracy', 0)
                            print(f"   {i}. Accuracy: {accuracy:.3f}")
                            if 'chunk_size' in config:
                                print(f"      chunk_size={config.get('chunk_size')}, overlap={config.get('overlap')}")
                            elif 'model_name' in config:
                                print(f"      {config.get('model_name')}, chunk_size={config.get('chunk_size')}")
                            elif 'fusion_strategy' in config:
                                print(f"      {config.get('fusion_strategy')}")
                    
                    print("-" * 50)
            else:
                print(f"⏳ Waiting for {filename} to be created...")
            
            time.sleep(5)  # Check every 5 seconds
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"❌ Error reading results: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_live_results() 