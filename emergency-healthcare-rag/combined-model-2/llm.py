import json
import ollama
from typing import Tuple, List, Dict

def load_topics_mapping():
    """Load topic name to ID mapping"""
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def classify_truth_and_topic_combined(statement: str, candidate_topics: List[Dict], context: str, model: str = 'llama3.1:8b') -> Tuple[int, int]:
    """
    Choose topic from candidates AND determine truth in single LLM call
    """
    
    # Create candidate list for prompt
    candidates_text = "\n".join([
        f"{topic['topic_id']}: {topic['topic_name']}" 
        for topic in candidate_topics
    ])
    
    # Enhanced medical prompt with more detailed instructions
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

Respond with ONLY two numbers separated by a comma: topic_id,truth_value
Example: 30,1"""

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