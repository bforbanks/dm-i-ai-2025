#!/usr/bin/env python3
"""
Main prediction interface for combined-model-2
Model-agnostic implementation with hybrid search and improved LLM
"""

from typing import Tuple
try:
    from .search import get_top_k_topics_with_context
    from .llm import classify_truth_and_topic_combined
except ImportError:
    from search import get_top_k_topics_with_context
    from llm import classify_truth_and_topic_combined

TOPIC_CANDIDATES_K = 8  # Number of topic candidates to consider

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
    # Get top-K topic candidates
    candidate_topics = get_top_k_topics_with_context(statement, k=TOPIC_CANDIDATES_K)
    
    # Use LLM to classify truth and topic in a single call (no context)
    statement_is_true, statement_topic = classify_truth_and_topic_combined(
        statement, candidate_topics, ""
    )
    
    return statement_is_true, statement_topic 