import json
import os
from pathlib import Path

def load_topics():
    """Load topic titles and their IDs"""
    topics_file = Path("data/topics.json")
    with open(topics_file, 'r') as f:
        topics = json.load(f)
    return topics

def create_prompt(statement: str, topics: dict) -> str:
    """Create a prompt for the LLM with all topic titles"""
    topic_list = "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(topics.keys())])
    
    prompt = f"""You are a medical expert. Given a statement about emergency healthcare, determine:
    1. Whether the statement is TRUE (1) or FALSE (0)
    2. Which medical topic it relates to (choose from the list below)

    Common patterns in FALSE statements include:
    - Incorrect numerical values (e.g., drug doses, lab cutoffs, time durations)
    - Reversed or false causation (e.g., misattributed physiological effects)
    - Overgeneralized treatment recommendations (ignoring contraindications or exceptions)
    - Misstated diagnostic criteria or staging definitions
    - Misleading statistics or prevalence claims
    - Anatomical or pathophysiological errors
    - Confusion between similar clinical terms or concepts

    Be skeptical of absolute claims with precise numbers, fixed durations, or universal recommendations.

    Statement: {statement}

    Available topics:
    {topic_list}

    Please respond in this exact format:
    Truth: [0 or 1]
    Topic: [topic name exactly as listed above]

    Example:
    Truth: 1
    Topic: Acute Coronary Syndrome"""
    return prompt

def predict(statement: str) -> tuple[bool, str]:
    """Predict truth and topic for a statement using LLM only"""
    import ollama
    
    # Load topics
    topics = load_topics()
    
    # Create prompt with all topic titles
    prompt = create_prompt(statement, topics)
    
    # Get LLM response
    response = ollama.chat(
        model='gemma3n:e4b',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    
    # Parse response
    try:
        result = response.message.content.strip()
        lines = result.split('\n')
        truth_line = [line for line in lines if line.startswith('Truth:')][0]
        topic_line = [line for line in lines if line.startswith('Topic:')][0]
        
        # Extract truth (0 or 1)
        truth = int(truth_line.split(':')[1].strip())
        
        # Extract topic name
        topic_name = topic_line.split(':')[1].strip()
        
        return bool(truth), topic_name
        
    except (IndexError, ValueError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Response was: {result}")
        # Fallback: return False and first topic
        return False, list(topics.keys())[0]

if __name__ == "__main__":
    # Test the model
    test_statement = "Coronary heart disease affects approximately 15.5 million people in the United States."
    truth, topic = predict(test_statement)
    print(f"Statement: {test_statement}")
    print(f"Truth: {truth}")
    print(f"Topic: {topic}") 