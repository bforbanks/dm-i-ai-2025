import importlib
import json
from typing import Tuple

# Model configuration - can be set by API or other calling scripts
# Default to match-and-choose-model-1, but this can be overridden
ACTIVE_MODEL = "match-and-choose-model-1"

def set_active_model(model_name: str):
    """Set the active model (called by API or other scripts)"""
    global ACTIVE_MODEL
    ACTIVE_MODEL = model_name

def get_active_model() -> str:
    """Get the currently active model"""
    return ACTIVE_MODEL

### CALL THE CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Model-agnostic prediction function. Routes to the currently active model.
    
    Args:
        statement (str): The medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    active_model = get_active_model()
    model_module = importlib.import_module(f"{active_model}.model")
    return model_module.predict(statement)

def match_topic(statement: str) -> int:
    """
    Simple keyword matching to find the best topic match.
    """
    # Load topics mapping
    with open('data/topics.json', 'r') as f:
        topics = json.load(f)
    
    statement_lower = statement.lower()
    best_topic = 0
    max_matches = 0
    
    for topic_name, topic_id in topics.items():
        # Extract keywords from topic name
        keywords = topic_name.lower().replace('_', ' ').replace('(', '').replace(')', '').split()
        
        # Count keyword matches in statement
        matches = sum(1 for keyword in keywords if keyword in statement_lower)
        
        if matches > max_matches:
            max_matches = matches
            best_topic = topic_id
    
    return best_topic
