#!/usr/bin/env python3
"""
LLM interaction for match-and-choose-model-1
Supports both truth-only and topic+truth classification based on threshold
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
    """Load topic name to ID mapping"""
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def generate_truth_only_prompt(statement: str, context: str) -> str:
    """Generate truth-only prompt (when above threshold - topic already decided)"""
    
    return f"""You are a medical expert. Determine if the following statement is TRUE or FALSE based on the provided medical context.

STATEMENT: {statement}

MEDICAL CONTEXT:
{context}

TASK: Determine if the statement is TRUE (1) or FALSE (0) based on the medical context above.

The chance of a statement being true or false is roughly 50/50. Previous tests indicate that you rate true too much, so lean towards false unless you know otherwise or the medical context supports true.
Common patterns in FALSE statements include:
    - Incorrect numerical values (e.g., drug doses, lab cutoffs, time durations)
    - Reversed or false causation (e.g., misattributed physiological effects)
    - Anatomical or pathophysiological errors

Respond with ONLY a single number: 1 for TRUE or 0 for FALSE.

NEVER SAY ANYTHING BUT A SINGLE NUMBER."""

def generate_topic_and_truth_prompt(statement: str, candidate_topics: List[Dict], context: str) -> str:
    """Generate topic+truth prompt (when below threshold - LLM chooses between candidates)"""
    
    # Format candidates with ranking and topic model confidence scores
    candidates_text = "\n".join([
        f"#{rank}: Topic {topic['topic_id']} - {topic['topic_name']} (Topic Model Score: {topic['score']:.3f})" 
        for rank, topic in enumerate(candidate_topics, 1)
    ])
    
    return f"""You are a medical expert. Analyze this statement and provide two determinations.

STATEMENT: {statement}

TOPIC MODEL RANKINGS (ordered by topic model confidence):
{candidates_text}

MEDICAL CONTEXT (chunks that matched these topics):
{context}

TASKS:
1. Choose the most relevant topic from the candidates above
2. Determine if the statement is TRUE (1) or FALSE (0) based on your medical knowledge and the context

IMPORTANT: The topic model found these candidates have very similar confidence scores (within threshold), which is why we need your expert medical judgment to choose between them. The topic model scores show how well each topic's content matched the statement keywords, but you should use your medical expertise to determine the most accurate topic.

The chance of a statement being true or false is roughly 50/50. Be skeptical of medical claims unless you are confident they are correct.
According to our stats, you rate true WAY too often. Thus, you should purposely try to rate more claims false. In other words, lean towards rating statements false.
Even if you think the current one is true, it might very well be false! Rate false often.

Common patterns in FALSE statements include:
    - Incorrect numerical values (e.g., drug doses, lab cutoffs, time durations)
    - Reversed or false causation (e.g., misattributed physiological effects)
    - Anatomical or pathophysiological errors

Respond with ONLY two numbers separated by a comma: topic_id,truth_value
    Examples: 
        YOUR ANSWER: "30,1"         corresponding to (topic 30, true)
        YOUR ANSWER: "45,0"         corresponding to (topic 45, false)
        
NEVER SAY ANYTHING BUT TWO NUMBERS SEPARATED BY A COMMA -- THE TOPIC, AND THE TRUTH BOOL."""

def classify_truth_only(statement: str, context: str, model: str = None) -> int:
    """
    Determine if a statement is true or false using rich medical context
    Used when gap > threshold (topic already decided by search)
    """
    if model is None:
        model = get_llm_model()
    
    # Log which model is being used
    model_info = get_model_info(model)
    print(f"Using LLM model: {model_info['name']} ({model_info['description']}) - Truth only")
    
    # Generate prompt
    prompt = generate_truth_only_prompt(statement, context)

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

def classify_topic_and_truth(statement: str, candidate_topics: List[Dict], context: str, model: str = None) -> Tuple[int, int]:
    """
    Choose topic from candidates AND determine truth in single LLM call
    Used when gap <= threshold (LLM needs to choose between similar candidates)
    """
    if model is None:
        model = get_llm_model()
    
    # Log which model is being used
    model_info = get_model_info(model)
    print(f"Using LLM model: {model_info['name']} ({model_info['description']}) - Topic + Truth")
    
    # Generate prompt
    prompt = generate_topic_and_truth_prompt(statement, candidate_topics, context)

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