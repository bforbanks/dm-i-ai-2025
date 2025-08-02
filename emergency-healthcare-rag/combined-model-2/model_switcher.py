#!/usr/bin/env python3
"""
Model switcher utility for combined-model-2
Easy switching between different LLM models
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import list_models, set_llm_model, get_llm_model, get_model_info

def print_available_models():
    """Print all available models with their information"""
    print("🤖 AVAILABLE LLM MODELS")
    print("=" * 80)
    
    models = list_models()
    current_model = get_llm_model()
    
    for model_id, info in models.items():
        status = "✅ CURRENT" if model_id == current_model else ""
        print(f"{model_id:20} | {info['parameters']:8} | {info['reasoning']:10} | {info['description']} {status}")
    
    print("=" * 80)
    print(f"Current model: {current_model}")
    print(f"To switch models: python model_switcher.py <model_id>")
    print(f"Example: python model_switcher.py deepseek-r1:32b")

def switch_model(model_id: str):
    """Switch to a different model"""
    models = list_models()
    
    if model_id not in models:
        print(f"❌ Error: Model '{model_id}' not found in available models")
        print("\nAvailable models:")
        for mid in models.keys():
            print(f"  - {mid}")
        return False
    
    # Set the model
    set_llm_model(model_id)
    
    # Show model info
    info = get_model_info(model_id)
    print(f"\n✅ Successfully switched to: {info['name']}")
    print(f"📊 Parameters: {info['parameters']}")
    print(f"🧠 Reasoning: {info['reasoning']}")
    print(f"📝 Description: {info['description']}")
    
    return True

def main():
    """Main function for model switching"""
    if len(sys.argv) == 1:
        # No arguments - show available models
        print_available_models()
    elif len(sys.argv) == 2:
        # One argument - switch to model
        model_id = sys.argv[1]
        if model_id in ['-h', '--help', 'help']:
            print("Model Switcher for Combined Model 2")
            print("Usage:")
            print("  python model_switcher.py                    # List available models")
            print("  python model_switcher.py <model_id>         # Switch to model")
            print("  python model_switcher.py help               # Show this help")
            print("\nExamples:")
            print("  python model_switcher.py gemma3:27b")
            print("  python model_switcher.py deepseek-r1:32b")
            print("  python model_switcher.py llama3.1:70b")
        else:
            switch_model(model_id)
    else:
        print("❌ Error: Too many arguments")
        print("Usage: python model_switcher.py [model_id]")

if __name__ == "__main__":
    main() 