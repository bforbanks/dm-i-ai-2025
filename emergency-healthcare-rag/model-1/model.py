from typing import Tuple
from .search import get_top_k_topics_with_context, get_targeted_context_for_topic, get_multi_topic_context
from .llm import classify_truth_only, classify_topic_and_truth_combined

# HYPERPARAMETER: Number of topic candidates to consider
TOPIC_CANDIDATES_K = 1  # Set to 1 for pure semantic, >1 for LLM selection
# K=1 should be much faster - test this first on UCloud!

### CALL THE CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    
    Configurable approach based on TOPIC_CANDIDATES_K:
    - K=1: Pure semantic topic selection + LLM truth classification (fastest)
    - K>1: LLM selects from top-K topics + determines truth in single call
    
    Args:
        statement (str): The medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    if TOPIC_CANDIDATES_K == 1:
        # Fast path: Pure semantic topic selection + separate truth classification
        candidate_topics = get_top_k_topics_with_context(statement, k=1)
        statement_topic = candidate_topics[0]['topic_id']
        
        # Get targeted context for the chosen topic
        context = get_targeted_context_for_topic(statement, statement_topic, max_chars=1500)
        
        # Truth classification with targeted context
        statement_is_true = classify_truth_only(statement, context)
        
        return statement_is_true, statement_topic
    
    else:
        # LLM path: Get top-K candidates, let LLM choose topic + determine truth
        candidate_topics = get_top_k_topics_with_context(statement, k=TOPIC_CANDIDATES_K)
        
        # Get context from all candidates
        context = get_multi_topic_context(candidate_topics, max_chars=1500)
        
        # Single LLM call for both topic selection and truth determination
        statement_is_true, statement_topic = classify_topic_and_truth_combined(
            statement, candidate_topics, context
        )
        
        return statement_is_true, statement_topic