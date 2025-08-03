#!/usr/bin/env python3
"""
LLM interaction for separated-models-1
Focused on truth determination using rich context
"""

import json
import os
import ollama
from typing import Tuple, List, Dict

try:
    from .config import get_llm_model, get_model_info
except ImportError:
    from config import get_llm_model, get_model_info

def generate_truth_prompt(statement: str, context: str) -> str:
    """Generate the truth determination prompt with rich context"""
    
    return f"""You are a medical expert. Determine if the following statement is TRUE or FALSE based on the provided medical context.

STATEMENT: {statement}

MEDICAL CONTEXT:
{context}

TASK: Determine if the statement is TRUE (1) or FALSE (0) based on the medical context above.

The chance of a statement being true or false is roughly 50/50. Be skeptical of medical claims unless they are clearly supported by the context.
In other words, lean towards rating statements false unless the context provides strong evidence they are correct.

Respond with ONLY a single number: 1 for TRUE or 0 for FALSE.

NEVER SAY ANYTHING BUT A SINGLE NUMBER."""

def classify_truth_only(statement: str, context: str, model: str = None) -> int:
    """
    Determine if a statement is true or false using rich medical context
    """
    if model is None:
        model = get_llm_model()
    
    # Log which model is being used
    model_info = get_model_info(model)
    print(f"Using LLM model: {model_info['name']} ({model_info['description']})")
    
    # Generate prompt
    prompt = generate_truth_prompt(statement, context)

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Parse response
        response_text = response['message']['content'].strip()
        
        # Handle different response formats
        if response_text in ['0', '1']:
            return int(response_text)
        
        # Fallback: try to extract number from response
        import re
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            return int(numbers[0])
            
        # Default fallback
        print(f"Warning: Could not parse response: {response_text}")
        return 0  # Default to false for safety
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return 0  # Default to false for safety 