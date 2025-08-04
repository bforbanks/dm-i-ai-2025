#!/usr/bin/env python3
"""
Configuration for separated-models-2
Model-agnostic LLM selection and settings
"""

import os
from typing import Dict, Any

class Config:
    # Default LLM model
    DEFAULT_LLM_MODEL = "gemma3n:e4b"  # Efficient model for local testing
    
    # Available models with descriptions
    AVAILABLE_MODELS = {
        "gemma3:27b": {
            "name": "Gemma 3 27B",
            "description": "Current, most capable model for single GPU",
            "size": "27B parameters"
        },
        "llama3.1:8b": {
            "name": "Llama 3.1 8B", 
            "description": "Fast, efficient model for local testing",
            "size": "8B parameters"
        },
        "gemma3n:e4b": {
            "name": "Gemma 3N E4B",
            "description": "Efficient 4B parameter model for local testing",
            "size": "4B parameters"
        },
        "llama3.1:12b": {
            "name": "Llama 3.1 12B",
            "description": "Balanced performance and speed",
            "size": "12B parameters"
        }
    }

def get_llm_model() -> str:
    """Get the LLM model name from environment variable or default"""
    return os.getenv('LLM_MODEL', Config.DEFAULT_LLM_MODEL)

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model"""
    return Config.AVAILABLE_MODELS.get(model_name, {
        "name": model_name,
        "description": "Unknown model",
        "size": "Unknown"
    })

def list_available_models():
    """List all available models"""
    print("Available LLM Models:")
    print("=" * 50)
    for model_id, info in Config.AVAILABLE_MODELS.items():
        current = " (CURRENT)" if model_id == get_llm_model() else ""
        print(f"{model_id}{current}")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
        print()

def set_llm_model(model_name: str):
    """Set the LLM model via environment variable"""
    if model_name not in Config.AVAILABLE_MODELS:
        print(f"Warning: {model_name} not in known models list")
    
    os.environ['LLM_MODEL'] = model_name
    print(f"LLM model set to: {model_name}") 