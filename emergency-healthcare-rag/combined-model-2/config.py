#!/usr/bin/env python3
"""
Configuration for combined-model-2
Model-agnostic settings and defaults
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for combined-model-2"""
    
    # LLM Model Configuration
    DEFAULT_LLM_MODEL = "gemma3:27b"
    
    # Available models for easy switching
    AVAILABLE_MODELS = {
        "gemma3:27b": {
            "name": "Gemma 3 27B",
            "description": "Current, most capable model for single GPU",
            "parameters": "27.4B",
            "context_window": "128K",
            "reasoning": "Excellent"
        },
        "gemma3:12b": {
            "name": "Gemma 3 12B", 
            "description": "Good balance of performance and speed",
            "parameters": "12B",
            "context_window": "128K",
            "reasoning": "Good"
        },
        "deepseek-r1:32b": {
            "name": "DeepSeek R1 32B",
            "description": "Specialized for reasoning tasks",
            "parameters": "32B", 
            "context_window": "128K",
            "reasoning": "Excellent"
        },
        "llama3.1:8b": {
            "name": "Llama 3.1 8B",
            "description": "Good reasoning for smaller size",
            "parameters": "8B",
            "context_window": "8K",
            "reasoning": "Good"
        },
        "llama3.1:70b": {
            "name": "Llama 3.1 70B",
            "description": "Maximum reasoning capability",
            "parameters": "70B",
            "context_window": "8K", 
            "reasoning": "Excellent"
        }
    }
    
    @classmethod
    def get_llm_model(cls) -> str:
        """Get the LLM model from environment variable or use default"""
        return os.getenv('LLM_MODEL', cls.DEFAULT_LLM_MODEL)
    
    @classmethod
    def get_model_info(cls, model_name: str = None) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_name is None:
            model_name = cls.get_llm_model()
        return cls.AVAILABLE_MODELS.get(model_name, {
            "name": model_name,
            "description": "Custom model",
            "parameters": "Unknown",
            "context_window": "Unknown",
            "reasoning": "Unknown"
        })
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available models with their information"""
        return cls.AVAILABLE_MODELS
    
    @classmethod
    def set_llm_model(cls, model_name: str) -> None:
        """Set the LLM model via environment variable"""
        os.environ['LLM_MODEL'] = model_name
        print(f"LLM model set to: {model_name}")
        print(f"Model info: {cls.get_model_info(model_name)}")

# Convenience functions
def get_llm_model() -> str:
    """Get the current LLM model"""
    return Config.get_llm_model()

def set_llm_model(model_name: str) -> None:
    """Set the LLM model"""
    Config.set_llm_model(model_name)

def list_models() -> Dict[str, Dict[str, Any]]:
    """List available models"""
    return Config.list_available_models()

def get_model_info(model_name: str = None) -> Dict[str, Any]:
    """Get information about a model"""
    return Config.get_model_info(model_name) 