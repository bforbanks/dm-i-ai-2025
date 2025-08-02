#!/usr/bin/env python3
"""
Main prediction interface for combined-model-2
Model-agnostic implementation with hybrid search and improved LLM
"""

from typing import Tuple
try:
    from .search import get_top_k_topics_with_context, get_targeted_context_for_topic
    from .llm import classify_truth_and_topic_combined
except ImportError:
    from search import get_top_k_topics_with_context, get_targeted_context_for_topic
    from llm import classify_truth_and_topic_combined

TOPIC_CANDIDATES_K = 8  # Number of topic candidates to consider
MAX_CONTEXT_CHARS = 2500  # Maximum context length for LLM

def predict(statement: str) -> Tuple[int, int]:
    """
    Predict truth value and topic for a medical statement
    
    Args:
        statement: Medical statement to evaluate
        
    Returns:
        Tuple of (truth_value, topic_id) where:
        - truth_value: 1 if true, 0 if false
        - topic_id: Integer topic ID (0-114)
    """
    # Get top-K topic candidates with context
    candidate_topics = get_top_k_topics_with_context(statement, k=TOPIC_CANDIDATES_K)
    
    # Get targeted context for the top candidate
    context = get_targeted_context_for_topic(statement, candidate_topics[0]['topic_id'], max_chars=MAX_CONTEXT_CHARS)
    
    # Use LLM to classify truth and topic in a single call
    statement_is_true, statement_topic = classify_truth_and_topic_combined(
        statement, candidate_topics, context
    )
    
    return statement_is_true, statement_topic 