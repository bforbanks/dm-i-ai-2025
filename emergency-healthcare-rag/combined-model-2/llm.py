#!/usr/bin/env python3
"""
LLM interaction for combined-model-2
Model-agnostic implementation for truth and topic classification
"""

import json
import os
import ollama
from typing import Tuple, List, Dict
from .config import get_llm_model, get_model_info

def load_topics_mapping():
    with open('data/topics.json', 'r') as f:
        return json.load(f)

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
    
    candidates_text = "\n".join([
        f"{topic['topic_id']}: {topic['topic_name']}" 
        for topic in candidate_topics
    ])
    
    prompt = f"""You are a medical expert specializing in emergency healthcare. Analyze this medical statement and provide two determinations:

STATEMENT: {statement}

RELEVANT MEDICAL CONTEXT:
{context}

TOPIC CANDIDATES (choose the most relevant):
{candidates_text}

TASKS:
1. Choose which topic (0-114) the statement best relates to from the candidates above
2. Determine if the statement is TRUE (1) or FALSE (0) based on the medical context

IMPORTANT GUIDELINES:
- Consider medical terminology, treatment protocols, and clinical guidelines
- Pay attention to specific medical conditions, medications, and procedures
- Base your truth determination on the provided medical context
- Choose the topic that most closely matches the medical content of the statement
- For medical statements, prioritize accuracy over speed
- Consider diagnostic criteria, treatment protocols, and clinical guidelines
- Look for specific medical terminology and procedures mentioned
- Be skeptical of medical claims - verify against the provided context

Respond with ONLY two numbers separated by a comma: topic_id,truth_value
Example: 30,1"""

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