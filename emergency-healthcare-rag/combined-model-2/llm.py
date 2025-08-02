#!/usr/bin/env python3
"""
LLM interaction for combined-model-2
Model-agnostic implementation for truth and topic classification
"""

import json
import os
import ollama
from typing import Tuple, List, Dict

try:
    from .config import get_llm_model, get_model_info
except ImportError:
    from config import get_llm_model, get_model_info

def load_topics_mapping():
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def generate_classification_prompt(statement: str, candidate_topics: List[Dict], context: str) -> str:
    """Generate the classification prompt"""
    candidates_text = "\n".join([
        f"{topic['topic_id']}: {topic['topic_name']}" 
        for topic in candidate_topics
    ])
    
    return f"""You are a medical expert. Analyze this statement and provide two determinations.

STATEMENT: {statement}

TOPIC CANDIDATES (ranked by relevance - prefer higher ranked ones):
{candidates_text}

TASKS:
1. Choose the most relevant topic from the candidates above (prefer higher ranked ones)
2. Determine if the statement is TRUE (1) or FALSE (0) based on your medical knowledge

The chance of a statement being true or false is roughly 50/50. Be skeptical of medical claims unless you are confident they are correct.
In other words, lean towards rating statements false.

Respond with ONLY two numbers separated by a comma: topic_id,truth_value
    Examples: 

        YOUR ANSWER: "30,1"         corresponding to (topic 30, true)
        YOUR ANSWER: "45,0"         corresponding to(topic 45, false)
        
        
    NEVER SAY ANYTHING BUT TWO NUMBERS SEPERATED BY A COMMA -- THE TOPIC, AND THE TRUTH BOOL."""

def classify_truth_and_topic_combined(statement: str, candidate_topics: List[Dict], context: str, model: str = None) -> Tuple[int, int]:
    """
    Choose topic from candidates AND determine truth in single LLM call
    Model-agnostic implementation that works with any Ollama model
    """
    if model is None:
        model = get_llm_model()
    
    # Log which model is being used
    model_info = get_model_info(model)
    print(f"Using LLM model: {model_info['name']} ({model_info['description']})")
    
    # Generate prompt
    prompt = generate_classification_prompt(statement, candidate_topics, context)

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        # Parse response
        response_text = response['message']['content'].strip()
        
        # Handle different response formats
        if ',' in response_text:
            parts = response_text.split(',')
            if len(parts) >= 2:
                topic_id = int(parts[0].strip())
                truth_value = int(parts[1].strip())
                return truth_value, topic_id
        
        # Fallback: try to extract numbers from response
        import re
        numbers = re.findall(r'\d+', response_text)
        if len(numbers) >= 2:
            topic_id = int(numbers[0])
            truth_value = int(numbers[1])
            return truth_value, topic_id
            
        # Default fallback
        print(f"Warning: Could not parse response: {response_text}")
        return 1, candidate_topics[0]['topic_id']
        
    except Exception as e:
        print(f"Error calling LLM: {e}")
        # Fallback to first candidate topic and assume true
        return 1, candidate_topics[0]['topic_id'] 