from typing import Tuple
try:
    from .search import get_top_k_topics_with_context, get_targeted_context_for_topic
    from .llm import classify_truth_and_topic_combined
except ImportError:
    from search import get_top_k_topics_with_context, get_targeted_context_for_topic
    from llm import classify_truth_and_topic_combined

# HYPERPARAMETER: Number of topic candidates to consider
TOPIC_CANDIDATES_K = 8  # Increased from 3 for better coverage

### CALL THE CUSTOM MODEL VIA THIS FUNCTION ###
def predict(statement: str) -> Tuple[int, int]:
    """
    Predict both binary classification (true/false) and topic classification for a medical statement.
    
    Enhanced approach using hybrid search and single LLM call:
    - Hybrid search (BM25 + Vector) to get top-K candidates
    - Single LLM call to classify both truth and topic from candidates
    
    Args:
        statement (str): The medical statement to classify
        
    Returns:
        Tuple[int, int]: (statement_is_true, statement_topic)
            - statement_is_true: 1 if true, 0 if false
            - statement_topic: topic ID from 0-114
    """
    # Get top-K topic candidates using hybrid search
    candidate_topics = get_top_k_topics_with_context(statement, k=TOPIC_CANDIDATES_K)
    
    # Get targeted context for the chosen topic
    context = get_targeted_context_for_topic(statement, candidate_topics[0]['topic_id'], max_chars=2500)
    
    # Single LLM call for both topic selection and truth determination
    statement_is_true, statement_topic = classify_truth_and_topic_combined(
        statement, candidate_topics, context
    )
    
    return statement_is_true, statement_topic 