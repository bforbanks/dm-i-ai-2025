#!/usr/bin/env python3
"""
Model switcher utility for separated-models-1
"""

import sys
import os
from pathlib import Path

# Add current directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from config import list_available_models, set_llm_model

def print_available_models():
    """Print all available models"""
    list_available_models()

def switch_model(model_name: str):
    """Switch to a different model"""
    set_llm_model(model_name)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python model_switcher.py list")
        print("  python model_switcher.py switch <model_name>")
        return
    
    command = sys.argv[1]
    
    if command == "list":
        print_available_models()
    elif command == "switch":
        if len(sys.argv) < 3:
            print("Error: Please specify a model name")
            print("Example: python model_switcher.py switch llama3.1:8b")
            return
        model_name = sys.argv[2]
        switch_model(model_name)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: list, switch")

if __name__ == "__main__":
    main() 