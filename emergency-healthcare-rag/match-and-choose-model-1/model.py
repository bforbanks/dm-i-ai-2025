#!/usr/bin/env python3
"""
Main prediction interface for match-and-choose-model-1
Threshold-based decision making between topic model and LLM choice
"""

from typing import Tuple, List, Dict, Union
try:
    from .topic_model import get_top_topics_with_scores, get_rich_context_for_statement
    from .llm import classify_truth_only, classify_topic_and_truth
    from .config import get_threshold, get_config_summary
except ImportError:
    from topic_model import get_top_topics_with_scores, get_rich_context_for_statement
    from llm import classify_truth_only, classify_topic_and_truth
    from config import get_threshold, get_config_summary

def get_candidates_context(statement: str, candidate_topics: List[Dict], max_chars: int = 2500) -> str:
    """
    Build context from multiple candidate topics for LLM to choose between
    """
    context_parts = []
    
    for topic in candidate_topics:
        topic_context = get_rich_context_for_statement(statement, topic['topic_id'], max_chars // len(candidate_topics))
        if topic_context:
            context_parts.append(f"--- TOPIC {topic['topic_id']}: {topic['topic_name']} (Score: {topic['score']:.2f}) ---\n{topic_context}")
    
    return "\n\n".join(context_parts)

def predict(statement: str, threshold: Union[float, str] = None) -> Tuple[int, int]:
    """
    Predict truth value and topic for a medical statement using threshold-based approach
    
    Args:
        statement: Medical statement to evaluate
        threshold: Optional threshold override. If None, uses config default.
                  'NA' means always use separated approach (topic model + truth LLM)
                  Float means use gap threshold for decision making
        
    Returns:
        Tuple of (truth_value, topic_id) where:
        - truth_value: 1 if true, 0 if false
        - topic_id: Integer topic ID (0-114)
    """
    # Get threshold from parameter or config
    if threshold is None:
        threshold = get_threshold()
    
    # Get configuration summary for logging
    config = get_config_summary()
    print(f"🎯 Match-and-Choose Model - Threshold: {threshold}, Model: {config['model_info']['name']}")
    
    # Get top topics with scores
    top_topics = get_top_topics_with_scores(statement, top_k=5)
    
    if len(top_topics) == 0:
        print("Warning: No topics found, defaulting to topic 0")
        return 0, 0
    
    if len(top_topics) == 1:
        print(f"Only one topic found: {top_topics[0]['topic_id']}, using separated approach")
        # Only one topic, use separated approach
        topic_id = top_topics[0]['topic_id']
        context = get_rich_context_for_statement(statement, topic_id)
        truth_value = classify_truth_only(statement, context)
        return truth_value, topic_id
    
    # Calculate gap between 1st and 2nd topic scores
    first_score = top_topics[0]['score']
    second_score = top_topics[1]['score']
    gap = first_score - second_score
    
    print(f"Gap analysis: 1st={first_score:.3f}, 2nd={second_score:.3f}, gap={gap:.3f}")
    
    # Decision logic based on threshold
    if threshold == 'NA':
        # Always use separated approach (topic model decides, LLM only does truth)
        print("🔧 Using SEPARATED approach (threshold=NA)")
        topic_id = top_topics[0]['topic_id']
        context = get_rich_context_for_statement(statement, topic_id)
        truth_value = classify_truth_only(statement, context)
        return truth_value, topic_id
        
    elif gap > threshold:
        # High confidence in topic model's 1st pick - use separated approach
        print(f"🔧 Using SEPARATED approach (gap {gap:.3f} > threshold {threshold})")
        topic_id = top_topics[0]['topic_id']
        context = get_rich_context_for_statement(statement, topic_id)
        truth_value = classify_truth_only(statement, context)
        return truth_value, topic_id
        
    else:
        # Low confidence gap - let LLM choose between candidates
        print(f"🤖 Using COMBINED approach (gap {gap:.3f} ≤ threshold {threshold})")
        
        # Get all topics within threshold of the best score
        # Only include topics where: score >= (1st_score - threshold)
        first_score = top_topics[0]['score']
        threshold_score = first_score - threshold
        
        candidate_topics = []
        for topic in top_topics:
            if topic['score'] >= threshold_score:
                candidate_topics.append(topic)
            else:
                break  # Scores are sorted, so we can stop once below threshold
        
        # Apply hard limit of 5 to keep prompt manageable
        candidate_topics = candidate_topics[:5]
        
        # Build context from all candidate topics
        context = get_candidates_context(statement, candidate_topics)
        
        # Let LLM choose topic and determine truth
        print(f"LLM choosing between {len(candidate_topics)} candidates within threshold")
        truth_value, topic_id = classify_topic_and_truth(statement, candidate_topics, context)
        
        print(f"LLM chose topic {topic_id} with truth value {truth_value}")
        return truth_value, topic_id

def predict_with_details(statement: str, threshold: Union[float, str] = None) -> Dict:
    """
    Predict with detailed decision information for analysis
    
    Returns:
        Dict containing prediction, decision rationale, and intermediate values
    """
    # Get threshold
    if threshold is None:
        threshold = get_threshold()
    
    # Get top topics
    top_topics = get_top_topics_with_scores(statement, top_k=5)
    
    if len(top_topics) < 2:
        gap = 999 if len(top_topics) == 1 else -1
        approach = "separated (insufficient topics)"
    elif threshold == 'NA':
        gap = top_topics[0]['score'] - top_topics[1]['score']
        approach = "separated (threshold=NA)"
    else:
        gap = top_topics[0]['score'] - top_topics[1]['score']
        approach = "separated" if gap > threshold else "combined"
    
    # Make prediction
    truth_value, topic_id = predict(statement, threshold)
    
    return {
        'prediction': {
            'truth_value': truth_value,
            'topic_id': topic_id
        },
        'decision_info': {
            'threshold': threshold,
            'gap': gap,
            'approach_used': approach,
            'top_topics': top_topics[:3],  # Show top 3 for analysis
            'config': get_config_summary()
        }
    }