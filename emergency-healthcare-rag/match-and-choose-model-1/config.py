#!/usr/bin/env python3
"""
Configuration for match-and-choose-model-1
Model-agnostic LLM selection and threshold settings
"""

import os
from typing import Dict, Any, Union

class Config:
    # Default LLM model
    DEFAULT_LLM_MODEL = "gemma3:27b"  # Most capable model for the match-and-choose approach
    
    # Default threshold for gap-based decision making
    DEFAULT_THRESHOLD = 0  # Use LLM when gap between 1st and 2nd topic is <= 0
    
    # Available models with descriptions
    AVAILABLE_MODELS = {
        "gemma3:27b": {
            "name": "Gemma 3 27B",
            "description": "Most capable model for match-and-choose decisions",
            "size": "27B parameters"
        },
        "llama3.1:8b": {
            "name": "Llama 3.1 8B", 
            "description": "Fast, efficient model for testing",
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

def get_threshold() -> Union[float, str]:
    """Get the threshold from environment variable or default"""
    threshold_str = os.getenv('THRESHOLD', str(Config.DEFAULT_THRESHOLD))
    
    # Handle special case 'NA' for no threshold (always use separated approach)
    if threshold_str.upper() == 'NA':
        return 'NA'
    
    try:
        return float(threshold_str)
    except ValueError:
        print(f"Warning: Invalid threshold '{threshold_str}', using default {Config.DEFAULT_THRESHOLD}")
        return Config.DEFAULT_THRESHOLD

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

def set_threshold(threshold: Union[float, str]):
    """Set the threshold via environment variable"""
    os.environ['THRESHOLD'] = str(threshold)
    print(f"Threshold set to: {threshold}")

def get_config_summary():
    """Get a summary of current configuration"""
    return {
        'llm_model': get_llm_model(),
        'threshold': get_threshold(),
        'model_info': get_model_info(get_llm_model())
    }