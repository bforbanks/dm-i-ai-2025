import json
import ollama
from typing import Tuple

def load_topics_mapping():
    """Load topic name to ID mapping"""
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def classify_truth_only(statement: str, context: str, model: str = 'gemma3:12b') -> int:
    """Classify medical statement for truth only - much faster"""
    
    prompt = f"""You are a medical expert. Determine if this medical statement is true or false.

    STATEMENT: {statement}

    CONTEXT: {context}

    Respond with ONLY: 1 (if true) or 0 (if false)"""

    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    
    try:
        result = response.message.content.strip()
        if '1' in result:
            return 1
        elif '0' in result:
            return 0
        else:
            return 1  # Default to true if unclear
    except:
        return 1  # Default to true on error

def classify_topic_and_truth_combined(statement: str, candidate_topics: list, context: str, model: str = 'gemma3:12b') -> tuple:
    """Choose topic from candidates AND determine truth in single LLM call"""
    
    # Create candidate list for prompt
    candidates_text = "\n".join([
        f"{topic['topic_id']}: {topic['topic_name']}" 
        for topic in candidate_topics
    ])
    
    prompt = f"""You are a medical expert. Analyze this statement and provide two determinations:

            STATEMENT: {statement}

            RELEVANT CONTEXT:
            {context}

            TOPIC CANDIDATES:
            {candidates_text}

            TASKS:
            1. Choose which topic (0-114) the statement best relates to from the candidates above
            2. Determine if the statement is TRUE (1) or FALSE (0) based on the context

            Respond with ONLY two numbers separated by a comma: topic_id,truth_value
            Example: 78,1"""

    response = ollama.chat(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    
    try:
        result = response.message.content.strip()
        # Parse "topic_id,truth_value" format
        import re
        numbers = re.findall(r'\d+', result)
        if len(numbers) >= 2:
            topic_id = int(numbers[0])
            truth_value = int(numbers[1])
            
            # Validate topic_id is one of the candidates
            valid_ids = [topic['topic_id'] for topic in candidate_topics]
            if topic_id not in valid_ids:
                topic_id = candidate_topics[0]['topic_id']
            
            # Validate truth_value
            if truth_value not in [0, 1]:
                truth_value = 1
            
            return truth_value, topic_id
        
        # Fallback parsing - look for individual patterns
        if '1' in result or 'true' in result.lower():
            truth_value = 1
        else:
            truth_value = 0
            
        return truth_value, candidate_topics[0]['topic_id']
        
    except:
        return 1, candidate_topics[0]['topic_id']  # Default values