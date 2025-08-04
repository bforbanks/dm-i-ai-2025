#!/usr/bin/env python3
"""
Main prediction interface for separated-models-2
BM25-only search with optimized chunking
"""

from typing import Tuple
try:
    from .search import get_best_topic, get_rich_context_for_statement
    from .llm import classify_truth_only
except ImportError:
    from search import get_best_topic, get_rich_context_for_statement
    from llm import classify_truth_only

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
    # Step 1: BM25 search determines the topic
    topic_id = get_best_topic(statement)
    
    # Step 2: Get rich context from BM25 search for truth determination
    context = get_rich_context_for_statement(statement, topic_id)
    
    # Step 3: LLM determines truth using the context
    statement_is_true = classify_truth_only(statement, context)
    
    return statement_is_true, topic_id 