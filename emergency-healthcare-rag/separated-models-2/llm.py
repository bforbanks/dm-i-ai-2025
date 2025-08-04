#!/usr/bin/env python3
"""
LLM module for separated-models-2
Model-agnostic LLM interface using Ollama
"""

import os
import json
from typing import Dict, Any, Optional
from config import get_llm_model, get_model_info
import ollama

class LLMInterface:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or get_llm_model()
        self.model_info = get_model_info(self.model_name)
        
    def generate_response(self, prompt: str, max_tokens: int = 2048, temperature: float = 0.1) -> str:
        """
        Generate response using the configured LLM model
        """
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            return response['response'].strip()
            
        except Exception as e:
            print(f"Error generating response with {self.model_name}: {e}")
            return f"Error: Unable to generate response with {self.model_name}"
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model_name
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return self.model_info
    
    def set_model(self, model_name: str):
        """Change the model being used"""
        self.model_name = model_name
        self.model_info = get_model_info(model_name)
        print(f"Switched to model: {model_name}")

def create_llm_interface(model_name: Optional[str] = None) -> LLMInterface:
    """Factory function to create LLM interface"""
    return LLMInterface(model_name) 